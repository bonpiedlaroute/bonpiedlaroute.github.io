---
title: "I self-hosted Buzz and tried to make its agents actually work"
---

*Relay image `ghcr.io/block/buzz:main` (digest
`sha256:ee3264f4c578…`), desktop `v0.5.8`, macOS arm64, Docker 29.6.2 under
Colima.*

Block released [Buzz](https://github.com/block/buzz) in July under Apache-2.0: a
Nostr-based team workspace where AI agents are first-class participants with
their own keypairs and audit trails, rather than bots behind an API. It
self-hosts, which is the interesting part — a company's messages and code never
have to touch Block's servers.

**This is not an install guide.** Block already wrote one —
[Run your own Buzz relay](https://engineering.block.xyz/blog/run-your-own-buzz-relay) —
and it is good. Follow it.

This is what happened *after* I followed it: three days of pushing on the parts a
vendor has no reason to document. Specifically, the question a company actually
has to answer before adopting this, which is not "can I run a relay" but "will
the agents earn their keep, and what will go wrong on the way there".

Two things I did not expect:

- **Getting the agents to say anything took an evening**, and four completely different root causes all produced the same symptom: silence.
- **Once working, they were better than I expected.** A Claude agent and a GPT agent split a task across a chat channel and produced a result I could verify to the digit.

Everything below is measured, and every claim points at a file and line.

---

## The relay itself is not the hard part

Following the official guide, from nothing to a working deployment:

| Step | Time |
|---|---|
| `docker pull ghcr.io/block/buzz:main` (356 MB, multi-arch) | 28 s |
| Generate the owner keypair | 2 s |
| `./run.sh start` — postgres, redis, minio, minio-init, relay all healthy | 18 s |
| `./run.sh restart` | 17 s |

**4 minutes 30 of machine time**, no Rust toolchain, no build. Inside those 18
seconds the relay applied 54 tables of migrations, registered its community and
bootstrapped the owner. `/_readiness` returns `{"status":"ready"}`, and the
community and owner rows survive a restart. For a project this young, the
production Compose bundle is real, not a demo.

One thing worth saying loudly because the guide does not: **do not build from
source to try this.** `just setup` wants the Hermit toolchain and a full release
build of a 29-crate Rust workspace. The published image exists and covers arm64.
That distinction is worth about 40 minutes.

### Two places the repo contradicts the blog

The official guide is correct. The repository is not always consistent with it,
and if you start from the repo — which is what a self-hoster does — you will not
see the guide's warnings.

**The `127.0.0.1` rule exists only in the blog post.** The guide tells you to set
`RELAY_URL=ws://127.0.0.1:3000` and explicitly not `localhost`. Neither
`deploy/compose/.env.example` nor `deploy/compose/README.md` mentions this. The
example file ships `BUZZ_DOMAIN=buzz.example.com`, and the obvious localisation
is `localhost`. More on why that matters below — it is the single most expensive
mistake available here.

**`BUZZ_AUTO_MIGRATE` has four different defaults.**

| Surface | Default |
|---|---|
| Binary (`main.rs:189`) | `false` |
| `deploy/compose/compose.yml` fallback | `false` |
| `deploy/compose/.env.example` | `true` |
| Helm `values.yaml:383` | `true` |

The guide says to set it to `true`, which is right. But if you miss that, the
relay starts perfectly against an empty schema: readiness only checks Postgres
and Redis connectivity, never schema freshness, so the container reports healthy
and fails under load.

Also worth knowing: there are **two `.env.example` files and the wrong one is
easier to find.** `/.env.example` (254 lines) is development config and still
carries `TYPESENSE_API_KEY`/`TYPESENSE_URL` — search moved to Postgres FTS and no
Typesense container exists anywhere — plus a `BUZZ_RATE_LIMIT_*` block that
`ARCHITECTURE.md §9` states is not implemented. The deployment file is
`deploy/compose/.env.example`, 53 lines.

### Two things to change before production

**Your relay calls Block's push gateway by default.** `config.rs:849` defaults
`BUZZ_PUSH_GATEWAY_DELIVERY_URL` to `https://push.buzz.xyz/v1/deliveries/apns`,
and it appears in neither `.env.example`. If your reason for self-hosting is that
your data stays yours, an undeclared outbound dependency in the default config is
worth knowing about. Set it to the empty string to disable NIP-PL push entirely.

This is not hypothetical: the mobile app ships with working push, so a
self-hosted relay with default settings does route notification delivery through
Block's infrastructure. To their credit the design minimises what that costs
you — Block states the relay never sees the device token and the gateway never
sees public keys, message content or metadata — so this is a dependency question
rather than a privacy hole.

Pointing it at your own gateway is not currently an option either:
`crates/buzz-push-gateway/src/config.rs:101` refuses to start unless the public
delivery URL is *exactly* `push.buzz.xyz`, mirrored as a JSON-schema `const` in
the Helm chart. So today the choice is Block's gateway or no push at all.

**A clean boot logs alarming warnings.** With `BUZZ_GIT_CONFORMANCE_PROBE=true`
(the production default), startup emits a burst of:

```
WARN transport drop (pre-classification: socket/send failure)
     phase=if_match_race round=0 racer=5
```

That is the object-store conformance probe racing itself on purpose. It is
normal. It also reads exactly like a broken deployment, and it is the first thing
an operator sees.

---

## Joining your own relay: 4 minutes, and it is well made

This is the path every employee of every self-hosting company walks, so I walked
it: download the DMG, connect to my own relay, get admitted. **Four minutes** —
faster than plenty of SaaS onboarding with SSO.

I expected to find friction here and mostly did not, which is worth saying
plainly because the rest of this post is harder on the product.

![Buzz onboarding: create a new identity key, or use an existing key](/assets/img/identity.jpg)

The closed-relay path is handled properly. Point the client at a relay that
requires membership and you get told exactly what to do:

![Not a member yet: ask a relay admin to add you as a member, then come back and try again](/assets/img/access.png)

*"Ask a relay admin to add you as a member, then come back and try again"*, with
your npub below it and a copy button. That is the whole procedure in one
sentence. The retry genuinely re-checks against the server — I tested it in both
directions, removing my membership and adding it back — and the join screen makes
the same point before you even get there:

![Join a community: enter the invite link or community URL](/assets/img/community.jpg)

> Joining a private community? Some communities need the owner to add you before
> you can join. Copy your public ID and send it to the community owner.

One real caveat about distribution, though it is not on this path: the
**in-app** add-community dialog leads with "create a hosted community", which
provisions on Block's own infrastructure via `app.builderlab.xyz` — hardcoded at
`desktop/src-tauri/src/builderlab.rs:15`, with no env override. If you ship this
client to your own customers, the most prominent button in that dialog sends them
to someone else's service.

### The one thing that i think is wrong here

**The app tells you to keep a backup, then hides the button.**

The first choice is *"Create a new identity key"* vs *"Use an existing key"*. The
label is jargon — having just read the codebase, I still paused — but a "What's
an identity key?" link sits at the bottom of the screen, and the copy behind it
is good. It explains the model and credits the tradeoff honestly:

> Your identity belongs to you, not Buzz. There's no password to reset, and Buzz
> can't recover your key if you lose it. Keep a backup somewhere safe and never
> share it. Anyone with your key can act as you.

![What's an identity key? Buzz uses an identity key instead of a traditional account](/assets/img/identity_key.jpg)

That is the right thing to say. Acting on it is the problem. The backup step
during onboarding is skippable — I skipped it without noticing, while
specifically intending not to. Creating one afterwards means Settings → Profile →
expand *Identity* → the *Private key* row → click **Reveal**, which displays your
secret key on screen, and only then do "Create backup" and "Test backup" appear.

So the one action that protects you from permanent identity loss sits behind the
one action security training tells you never to take. A careful user will never
click Reveal, and will therefore never find the backup. Nothing prompts you later
either — and as measured further down, that omission is expensive.

Credit where it is due: the **Test backup** flow, once found, verifies that a
backup actually decrypts before you need it. Few tools do that. It is in the same
hiding place.

---

## Making the agents answer: four silent failures

I added the preset agents, Fizz and Honey, and talked to them. Nothing came back.
No error, no status, no "I am not set up yet" — silence in the channel, under
avatars that look entirely operational.

It took an evening, and the answer was four independent failures stacked on one
path. Each is defensible alone. Together they are why someone evaluating Buzz
concludes the agents do not work and leaves.

**The common thread is the thing worth fixing: every one of them surfaces as
nothing at all.**

### 1. An unconfigured agent looks exactly like a broken one

Both harness processes were running. Their environment states the problem
outright:

```json
BUZZ_ACP_SETUP_PAYLOAD={"agent_name":"Honey", …,
  "requirements":[{"field":"provider"},{"field":"model"}]}
```

No provider, no model — the "set up your agent harness" step I had skipped during
onboarding. The agents connect and listen; they have nothing to think with. That
behaviour is reasonable. Surfacing it as silence is not: the one thing a user
cannot distinguish is *not configured* from *broken*.

A subtler trap sits underneath. The presets are wired as a pair:

```
Fizz   BUZZ_ACP_RESPOND_TO=owner-only
Honey  BUZZ_ACP_RESPOND_TO=allowlist → <Fizz's pubkey>
```

Honey's allowlist holds *Fizz's* key, not mine. It is a sub-agent that only
answers Fizz, so messaging it directly returns nothing — by design, and nothing
in the channel UI conveys that.

### 2. `localhost` — why the official guide's one-line rule matters

The guide says to use `127.0.0.1` and not `localhost`, because "agents
canonicalize localhost to 127.0.0.1, so the URLs won't match". That is accurate
and it is the fix. Here is the mechanism, which nothing explains, and what it
looks like when you miss it.

The relay's tenant fence is strict. Only the registered host binds:

| `Host` | Result |
|---|---|
| `localhost:3000` | **101** |
| `127.0.0.1:3000` | 404 |
| `localhost`, `127.0.0.1`, `0.0.0.0:3000`, `relay.local:3000` | 404 |

No loopback folding anywhere on the relay side. `router.rs:265` takes the `HOST`
header verbatim, `normalize_host` only lowercases and strips `:80`/`:443` plus a
trailing dot, and `lookup_community_by_host` (`buzz-db/src/lib.rs:1195`) is an
exact `lower(host) = lower($1)` match. The relay is right, and deliberately so —
the surrounding comments describe host binding as a security boundary that
"never" yields a default tenant.

The desktop app connects with its configured URL. But the agents it spawns are
handed a *different* one:

```
BUZZ_ACP_DISPLAY_NAME=Fizz
BUZZ_RELAY_URL=ws://127.0.0.1:3000
```

The chain:

```
managed_agents/runtime_types.rs:22   relay_url = normalize_relay_url(relay_url)
managed_agents/runtime.rs:501        effective_relay_url = runtime_key.relay_url
managed_agents/runtime.rs:532        command.env("BUZZ_RELAY_URL", &effective_relay_url)
```

`normalize_relay_url` (`buzz-core/src/relay.rs:56`) folds every loopback spelling
to `127.0.0.1`. Its own doc comment states what that form is for:

> Connection code may retain the configured URL; this canonical form is for
> **identity**, receipts, status and deduplication.

So the app passes the identity-canonical form as the agent's *connection* URL,
against a contract written three lines above the function. Two normalizers with
opposite rules, and the failure lands on the user as an agent that never speaks.

Set `RELAY_URL=ws://127.0.0.1:3000` as the guide says, and use the same spelling
in the desktop app.

### 3. Two spellings, two tenants

Do not mix them. They register as separate communities:

```
host             | id
-----------------+--------------------------------------
localhost:3000   | 726fefec-f524-4cfe-ba6d-af868b15e44a
127.0.0.1:3000   | 51f05836-6a76-420c-b7e2-6e0c1610c577
```

A user on one and an agent on the other are in different tenants. Both connect
happily. Neither sees the other. Silence again.

### 4. Connecting is not joining

Aligned on one spelling, every agent connected for the first time —
`buzz_ws_connections_active` went 0 → 5 — and channels finally persisted
server-side. One last stall: the agents were attached to the relay while
`general` still had exactly one member, so they never saw the messages. Adding
Fizz to the channel was the final step.

Then it replied.

---

## Are the agents actually any good?

This is the question the official guide has no reason to answer, and the reason I
did all of the above. Yes — and it is the part that changed my mind about the
product.

I gave the agents a task with a **verifiable ground truth**, because "it replied"
is not evidence of anything: compare the repo's two `.env.example` files and post
a merged table. I already knew the answer.

I have configured Fizz to run Claude code on Claude Sonnet 5 and Bumble to run Codex on GPT-5.4-mini. Fizz read the
compose file, delegated the root file to Bumble by mentioning it in the channel,
waited, merged both halves, and posted a table with a per-row attribution column.

| | agent | truth |
|---|---|---|
| root file | 73 | 73 |
| compose file | 32 | 32 |
| in common | 7 | 7 |
| dev only | 66 | 66 |
| prod only | 25 | 25 |
| the 7 common names | — | identical |

Zero omissions, zero inventions, one duplicated line.

The detail that matters more than the totals: `BUZZ_RELAY_PRIVATE_KEY` is
*commented out* in the root file and live in the compose file. Both agents
counted it. Nobody told either of them whether commented variables were in
scope — two different models from two different vendors independently applied the
same convention, and their halves merged without contradiction.

Two other things here that are not available on others professional messaging app:

**It acted rather than chatted.** It read files, wrote `/tmp/env_compare.md`, and
posted through the native CLI:

```
buzz messages send --channel 7fdd5a94-… --reply-to a22d2ede… --content - < /tmp/env_compare.md
```

**It claimed work with an emoji.** Both agents reacted to my message to claim it,
the bubble showed them working, and each removed its reaction on finishing. A
coordination protocol expressed in the native grammar of a chat app. A Teams or Slack bot
posts "working on it…"; this reads as a colleague picking up a ticket.

**One caveat.** An earlier run stalled: I had mentioned the delegate in my own
message too, so it answered *me* while Fizz went on waiting for a reply to a
request that had already been overtaken. A race between delegator and delegate
when both are addressed at once. Removing my mention fixed it immediately.
Observed once; I have not tried to reproduce it deliberately.

**Verdict.** Two vendor-independent agents, each with its own keypair and audit
trail, split a task over a chat channel and produced a mutually consistent,
externally verifiable result. Slack has no equivalent, and neither does a bot
framework bolted onto one. The path to get there is brittle. The capability
underneath is not a demo.

---

## What I got wrong

For most of the first evening I was measuring the wrong relay.

Twelve days earlier I had built and run `target/debug/buzz-relay` to test a
relay-side fix I was contributing to Buzz, and never stopped it. It still held
`*:3000`. My Compose deployment started cleanly on top of it, `docker port`
reported `3000/tcp -> 0.0.0.0:3000` exactly as expected, and the container
stayed `healthy` for 23 hours — while being completely unreachable from the
host. No bind error, no warning, no degraded status anywhere.

The symptoms made no sense until I ran `lsof`. The relay database held three
events and zero channels while the desktop app cheerfully displayed seven
channels with recent messages. Both were true: the app was talking to the July
binary, I was querying the August container.

```bash
# the check that should have been first
lsof -nP -iTCP:3000 -sTCP:LISTEN
```

`curl localhost:3000/_liveness` returning `ok` proves a relay is up. It does
not prove it is *yours*. If you contribute to Buzz and also evaluate a
self-hosted deployment on the same machine, those two activities collide on
port 3000 — and under Colima the published port is served by an SSH forwarder,
which makes ownership even harder to see. Everything I concluded about host
binding before running that command was wrong.

One smaller correction: my first count of the root `.env.example` said 18
variables, because I excluded commented-out entries. The agents included them,
which is the better reading of "documented variables". They were right and I
was not.

---

## What this means if you are evaluating Buzz for a team

The relay is ready. The agents are genuinely differentiated and I would not have
believed that without measuring it. Three constraints decide whether it fits:

**Every seat needs a native install.** `web/` is five routes — an invite landing
page and a repo browser that is off by default — with no chat code at all. There
is no "just send them a link". Budget endpoint deployment for 100% of seats.

**Mobile exists, but it cannot stand alone.** Block shipped the app to
[both stores](https://engineering.block.xyz/blog/a-buzz-on-your-phone) with
working push notifications, and the design is careful — the relay never sees the
device token, and the push gateway never sees public keys or message content. But
the app requires pairing with a desktop install; Block says standalone mobile is
something they "could support" in future. So it does not relieve the previous
point, it compounds it: a mobile user needs a laptop first.

If you plan to fork the client rather than use Block's builds, note that mobile
is the one surface you cannot ship yourself — the release pipeline is private and
OSS CI cannot trigger it, and `mobile/pubspec.yaml` carries `publish_to: none`.

**Losing a laptop: what comes back, and what silently doubles.** There is no
server-side recovery, no escrow, no mnemonic — only that local NIP-49
`ncryptsec` file. So I deleted my identity and restored it from the backup, on a
relay with membership enforcement on, and compared the server state before and
after.

The identity recovery is exactly right. My membership survived, so I was
re-admitted with no admin action. All nine of my messages were still attributed
to me. Profile, channels and history came back untouched.

The agents do not come back — they come back *twice*.

```
before: 4 agent identities        after: 7
  Fizz    3fba3fee…                 Fizz    3fba3fee…  +  123abc0d…
  Bumble  24e0ae0e…                 Bumble  24e0ae0e…  +  898ab751…
  Honey   c86d0e6d…                 Honey   c86d0e6d…  +  404dbf98…

#Welcome members: 4 → 7   (one human, six bots)
```

The wipe destroys `agents/managed-agents.json`, so the presets are re-seeded with
fresh keypairs. The old agent identities stay registered server-side, still
flagged as agents, still holding their messages — five for the old Fizz, four for
the old Bumble — and still sitting in the channel as members.

That is worse than losing them, because it is invisible. You open the app, see
Fizz in the list, and assume continuity. In fact the previous Fizz's history is
attached to a key that no longer drives any process, and your channel holds two
same-named bots that are two different entities.

For a company the cost compounds: every replaced laptop doubles the agent roster,
and nothing prunes it. After three incidents you have nine entries called Fizz,
three of them live.

To be clear about what does *not* break: closing the app, disconnecting or
rebooting costs nothing. `relay_members` has no expiry column and nothing prunes
inactive members, so you come straight back in. An admin is only needed when
someone is explicitly removed, or arrives with a *different* pubkey — which is
what happens if they sign out and then create a new identity instead of restoring
one. Sign-out is gated behind two confirmations, so it is not a one-click
accident; but it is the only route that turns a returning employee into a
stranger.

---


