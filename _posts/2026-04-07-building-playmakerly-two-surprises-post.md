# Building Playmakerly: Two Things That Actually Surprised Me

---

For the past few months I've been building [Playmakerly](https://playmakerly.fr) — a Slack bot that uses AI to animate team predictions around football matches. Think office banter, but structured and powered by Claude Sonnet + LangGraph.

The stack: FastAPI, Celery, PostgreSQL, Redis, Next.js, and a LangGraph agent with tool-use for the conversational layer.

This post is about two things that caught me off guard during the build — not theoretical tradeoffs, but decisions that broke in production and forced me to think harder.

---

## 1/ My Celery scheduler was burning 576 API calls/day. With zero matches scheduled.

### The setup

Playmakerly needs live scores and match results to trigger its AI messages — post-match storytelling, leaderboard updates, rivalry banter. Since the football data API I use ([API-Football](https://www.api-football.com/)) doesn't offer webhooks, polling is the only option.

I set up Celery Beat with three recurring tasks:

```python
celery_app.conf.beat_schedule = {
    "check-match-reminders": {
        "task": "app.tasks.match_tasks.send_match_reminders",
        "schedule": crontab(minute="*/30"),
    },
    "check-match-results": {
        "task": "app.tasks.match_tasks.send_match_results",
        "schedule": crontab(minute="*/15"),
    },
    "sync-live-scores": {
        "task": "app.tasks.match_tasks.sync_live_scores",
        "schedule": crontab(minute="*/5"),
    },
    "sync-upcoming-matches": {
        "task": "app.tasks.sync_tasks.sync_upcoming_matches",
        "schedule": crontab(hour=3, minute=0),
    },
}
```

Standard pattern. Fire and forget.

### What went wrong

The tasks called the external API unconditionally — no check first, just straight to the HTTP call.

Result: **100+ unnecessary API calls in a single day**, with zero matches in the database. The daily quota was hit. Real Champions League fixtures stopped syncing.

To make it worse, my local Docker setup with Celery Beat had been running in the background for two weeks without me noticing — quietly consuming quota the whole time.

### The fix

One SQL check before every external call:

```python
# Before — straight API call, no questions asked
async def _sync_live_scores():
    client = FootballAPIClient()
    live_fixtures = await client.get_live_matches_all()  # 2 API calls every 5 min
    # ...

# After — one local DB check before any external call
async def _sync_live_scores():
    async with get_celery_db() as db:
        today_start = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=UTC)
        today_end = today_start + timedelta(days=1)

        result = await db.execute(
            select(Match).where(
                Match.status.in_([MatchStatus.SCHEDULED, MatchStatus.LIVE]),
                Match.start_time.between(today_start, today_end),
            )
        )
        if not result.scalars().all():
            logger.info("No matches today, skipping live score sync")
            return  # Zero API calls. Done.

    client = FootballAPIClient()
    live_fixtures = await client.get_live_matches_all()
    # ...
```

**Result: ~98% reduction in external API calls.** From 576/day to roughly 10-15 on match days only.

### The lesson

> When consuming a paid API via a scheduler, the first line of your task should always be: *"do I have a reason to make this call?"*

Idempotent tasks are good. Tasks that don't run when unnecessary are better.

A single local DB query costs ~1ms. An unnecessary external API call costs quota, money, and can block your real use cases.

---

## 2/ I didn't make everything agentic. On purpose.

### The temptation

When you're building with LLMs, the tempting move is to route everything through an agent. More flexible, more impressive to demo, feels more "AI-native."

For Playmakerly, every interaction involves Claude — post-match messages, reminders, leaderboard storytelling, @mention responses. I could have built one agent to handle all of it.

I didn't.

### Two modes, one clear boundary

Playmakerly runs two distinct AI modes:

**Mode 1 — Simple Claude prompts** for structured, event-driven content:
- Post-match storytelling ("PSG won 2-1, here's how the league reacts")
- Weekly reminders before a matchday
- Leaderboard updates with rivalry context

```python
class AIService:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = "claude-sonnet-4-20250514"

    async def generate_message(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        system_prompt = self._build_system_prompt(context)
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
```

The context is already fully structured (scores, predictions, streaks, rivalries). There's no ambiguity about what to say — just *how* to say it well. A prompt template is enough. An agent would add latency and token cost for zero benefit.

**Mode 2 — LangGraph agent with tools** for open @mention questions:

```python
async def generate_response(state: AgentState, db: AsyncSession) -> dict[str, Any]:
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=300, temperature=0.7)

    tools = create_tools(db, league_id=state["league_id"], user_id=state["user_id"])
    llm_with_tools = llm.bind_tools(tools)
    # tools: get_leaderboard, get_upcoming_matches, get_user_stats, get_head_to_head, get_user_streak

    messages = [
        SystemMessage(content=_build_system_prompt(state)),
        HumanMessage(content=state["user_message"]),
    ]

    response = await llm_with_tools.ainvoke(messages)
```

When someone asks *"@playmakerly who's been the most consistent predictor this month?"*, the agent needs to:
1. Query the DB for prediction history
2. Calculate streaks and accuracy
3. Cross-reference with recent matches
4. Compose a contextual, personalized response

That's genuinely multi-step reasoning over unstructured input. An agent earns its cost here.

### The boundary

```
Structured data + known context  →  prompt template
Open question + unknown context  →  LangGraph agent
```

The agent caps at **3 tool-use iterations** to prevent infinite loops — enough for multi-step reasoning, not enough to burn your token budget or hit latency limits.

```python
    # Handle tool calls (max 3 iterations to avoid infinite loops)
    iterations = 0
    while response.tool_calls and iterations < 3:
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_fn = next((t for t in tools if t.name == tool_call["name"]), None)
            if tool_fn:
                tool_result = await tool_fn.ainvoke(tool_call["args"])
                messages.append({"role": "tool", "content": str(tool_result), "tool_call_id": tool_call["id"]})
        response = await llm_with_tools.ainvoke(messages)
        iterations += 1
```

### The lesson

> "All agentic" is tempting but expensive. The right pattern: ask yourself *"do I already know the context?"* If yes, a prompt template is enough. If no, reach for the agent.

Agents are powerful. They're also slower, costlier, and harder to debug. Use them where the open-endedness actually requires it.

---

## What's next

Playmakerly is live at [playmakerly.fr](https://playmakerly.fr), timed for the 2026 World Cup.

Next post: the Celery/asyncio impedance mismatch — why each task creates and destroys its own SQLAlchemy engine, and why that's actually the right call when your app is async but your workers aren't.

---

*Feedback welcome — try the bot at [playmakerly.fr](https://playmakerly.fr).*