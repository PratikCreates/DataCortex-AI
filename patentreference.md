# VIT IPR&TTCELL

## Invention Disclosure Format (IDF)-B

| Field | Value |
| :--- | :--- |
| Document No. | 02-IPR-R003 |
| Issue No./Date | 2/01.02.2024 |
| Amendment No./Date | 0/00.00.0000 |
| Draft Status | Professional technical disclosure draft for IPR/legal review |

---

## 1. Title of the Invention

**Cognitive Signal Router: A Privacy-Tiered System and Method for Hybrid Signal Extraction, Decision-State Routing, and Controlled Tool Execution**

---

## 2. Field / Area of Invention

The disclosure falls within artificial intelligence systems, mobile computing, information retrieval, task automation, and privacy-preserving personal data processing. The invention, referred to herein as the **Cognitive Signal Router**, concerns a computing architecture that receives user-specific information from multiple sources, extracts actionable signals, assigns those signals to finite operational states, and permits only controlled actions through typed tools and trust policies.

The system is intended for situations where important obligations, deadlines, or follow-up items are dispersed across personal records, communications, files, reminders, and prior context.

### Suggested IPC/CPC Classification

| Code | Relevance |
| :--- | :--- |
| G06F 16/90 | Information retrieval and semantic memory access |
| G06F 16/33 | Query processing across heterogeneous textual data |
| G06N 3/08 | Machine-learning-assisted extraction and ranking |
| G06F 21/62 | Privacy protection for personal data |
| G06Q 10/10 | Task, reminder, schedule, and workflow management |
| G06F 9/54 | Controlled inter-module tool execution |

---

## 3. Prior Patents and Publications from Literature

The following references identify relevant technical areas. The disclosed invention is distinguished by the combined control architecture: hybrid extraction, privacy-tiered memory, finite decision-state routing, and tool execution governed by scope and trust level.

| # | Reference | Relevant Teaching | Distinction |
| --- | --- | --- | --- |
| 1 | US20170118348A1, "Personalized reminders" | Reminder generation using urgency and availability | Does not disclose hybrid extraction from heterogeneous personal data followed by finite decision-state routing and controlled tool execution. |
| 2 | US20120016678A1, intelligent automated assistant | Conversational assistant with service invocation | Does not provide a privacy-tiered memory architecture or an explicit decision-state layer between retrieval and action. |
| 3 | US20220391803A1, AI-driven task management | Task planning and execution assistance | Does not combine deterministic extraction, structured output, local fallback, and trust-governed tool access. |
| 4 | US11282042B2, calendar conflict resolution | Calendar-specific conflict handling | Limited to calendar conflicts and does not address multi-source signal extraction or privacy-tiered memory. |
| 5 | Retrieval-Augmented Generation, Lewis et al. | Retrieval-augmented text generation | Teaches retrieval for generation, not operational state routing or controlled personal-data actions. |
| 6 | ReAct, Yao et al. | Reasoning and acting with language models | Provides a prompting pattern, not a scoped execution architecture with per-capability tool limits. |
| 7 | Memory-oriented language-model agent systems | Long-term context management | Does not disclose the claimed combination of private memory separation, hybrid extraction, state routing, and escalation-aware action control. |

---

## 4. Background and Technical Problem

Users often miss important items because the relevant signal is not stored as a ready-made task. It may be present in a message, document, note, reminder, calendar entry, or earlier conversation. Conventional task systems depend on manual entry. Conventional assistants may summarize information, but they commonly return unstructured text and do not provide a reliable control layer for deciding whether an item should be acted on, scheduled, monitored, or ignored.

The technical problem is therefore a control problem: how to transform mixed personal information into reliable operational decisions while preventing excessive data exposure and unsafe automated actions.

Existing systems do not adequately solve this problem because they generally lack one or more of the following:

- a privacy gate that separates private-local content from assistant-eligible memory;
- hybrid extraction using both deterministic rules and model-based inference;
- a finite state layer for routing extracted signals;
- a structured output contract suitable for deterministic user interfaces;
- tool-level authorization and confirmation rules; and
- local fallback behavior for time-sensitive use.

---

## 5. Summary of the Invention

The Cognitive Signal Router provides a system and method in which heterogeneous user-specific signals are normalized, privacy-classified, analyzed through hybrid extraction, routed into finite decision states, and converted into controlled actions only through authorized tools.

In one embodiment, the system includes:

1. an ingestion module that receives signals from personal information sources;
2. a privacy gate that separates private-local data from assistant-eligible data;
3. a hybrid extraction engine that identifies dates, obligations, entities, urgency, relevance, and action candidates;
4. a decision-state router that assigns each signal to an operational state;
5. a structured response generator that produces compact outputs with explanation and citations;
6. a bounded tool runtime that exposes only approved typed tools;
7. a trust policy that determines whether an action may run automatically, be suggested, or require confirmation; and
8. an action layer that creates or proposes tasks, reminders, alarms, calendar actions, notifications, or monitoring items.

The disclosed architecture is centered on the decision-state router. That router prevents raw retrieval or generative output from directly becoming an action. Instead, extracted evidence is first reduced to a controlled state such as `act_now`, `schedule`, `monitor`, or `ignore`.

---

## 6. Objectives of the Invention

The principal objectives are:

1. to identify actionable signals before the user manually converts them into tasks;
2. to reduce noisy assistant output into concise operational decisions;
3. to improve extraction reliability by combining deterministic and model-based methods;
4. to prevent private-local content from entering cloud memory or model retrieval;
5. to restrict assistant capabilities through typed tools and trust policies;
6. to provide source-grounded explanations for surfaced items;
7. to continue limited operation using local data and cached state when remote services are unavailable; and
8. to support escalation when an unresolved item approaches a relevant time threshold.

---

## 7. Working Principle

The invention operates as a controlled signal-processing pipeline.

![Figure 1. Cognitive Signal Router system architecture](docs/ip/figures/figure_1_cognitive_signal_router_architecture.png)

**Figure 1** illustrates the system architecture. Candidate signals are normalized, privacy-classified, analyzed, routed, and converted into bounded outputs. The key control point is the decision-state router, which sits before any tool execution.

### 7.1 Signal Normalization

Incoming information is converted into a candidate signal object. The object may include:

- source type;
- source reference;
- user scope;
- environment scope;
- timestamp;
- short content summary;
- extracted metadata; and
- processing status.

The normalized signal object allows different source types to be processed through a common downstream pipeline.

### 7.2 Privacy and Scope Gate

Before extraction or retrieval, the system assigns the signal to a processing class.

| Processing Class | Handling |
| :--- | :--- |
| Private-local | Stored and processed locally; excluded from cloud memory, embeddings, and model retrieval. |
| Smart memory | Eligible for scoped indexing and assistant retrieval. |
| Authorized external | Processed only within the authorized user and environment scope. |
| Local document | Parsed locally where possible; only bounded excerpts or extracted facts are exposed. |

The privacy gate is structural. Private-local content does not become a cloud memory object and is not available to the assistant retrieval path.

### 7.3 Hybrid Extraction

The extraction engine combines deterministic and model-based processing. Deterministic processing may include date parsing, pattern matching, entity recognition, source metadata analysis, duplicate checks, and table or document parsing. Model-based inference is used where the signal is semantically ambiguous or cannot be fully resolved by rules.

The two extraction paths are merged into an evidence record. The evidence record may include:

- detected deadline or time window;
- entity or subject;
- obligation or requested action;
- priority or urgency cue;
- source confidence;
- uncertainty flag; and
- suggested next operation.

### 7.4 Decision-State Routing

The decision-state router converts extracted evidence into a finite operational state.

| State | Meaning | Downstream Handling |
| :--- | :--- | :--- |
| `act_now` | The signal requires prompt attention | Surface prominently and prepare escalation or action proposal |
| `schedule` | A future action is appropriate | Create or propose a reminder, alarm, task, or calendar item |
| `monitor` | Relevant but uncertain or not yet actionable | Track, re-evaluate, or request confirmation |
| `ignore` | Duplicate, stale, low relevance, or non-actionable | Suppress from active surfaces |

![Figure 2. Cognitive Signal Router decision pipeline](docs/ip/figures/figure_2_cognitive_signal_router_pipeline.png)

**Figure 2** shows how extracted evidence passes through state assignment, structured output, and trust policy before a tool is allowed to act.

### 7.5 Structured Output Contract

The assistant response is constrained to a structured contract. A representative output includes:

```json
{
  "summary": "...",
  "decision_state": "act_now | schedule | monitor | ignore",
  "why_it_matters": "...",
  "key_items": [],
  "citations": [],
  "actions": []
}
```

The contract prevents long, unconstrained responses and enables deterministic rendering, source citation, and confirmation controls.

### 7.6 Controlled Tool Execution

The system exposes tools only through a bounded runtime. Tool access is determined by:

- the active capability;
- the user's authorization;
- the active environment;
- the permitted tool list;
- the data scope of the source; and
- the trust level of the requested action.

Representative tool categories include search, memory retrieval, document excerpting, table querying, task creation, reminder creation, calendar proposal, notification, and escalation.

### 7.7 Trust Policy

Actions are divided into three trust classes.

| Trust Class | Handling |
| :--- | :--- |
| `AUTO` | Read-only or non-side-effecting operations may run automatically. |
| `SUGGEST` | Low-risk write actions are prepared but require acceptance. |
| `CONFIRM` | External or high-impact actions require explicit confirmation. |

This prevents an assistant response from becoming an uncontrolled external action.

---

## 8. Detailed Description of Main Components

### 8.1 Ingestion Module

The ingestion module receives information from user-approved sources and converts it into a normalized signal object. The module is source-agnostic: it does not require all sources to use the same format, but it emits a common candidate structure for downstream processing.

### 8.2 Privacy-Tiered Memory

The memory subsystem separates private-local records from smart memory records. Private-local records remain outside cloud indexing, model retrieval, and assistant memory. Smart memory records are stored with user and environment scope and may be retrieved only through controlled tools.

### 8.3 Hybrid Extraction Engine

The extraction engine is designed for reliability. Deterministic rules handle explicit patterns and structured evidence. Model inference handles ambiguous natural-language context. The merged result is not directly executed; it is passed to the decision-state router.

### 8.4 Decision-State Router

The router receives extracted evidence and assigns a state. The state determines whether the system escalates, schedules, monitors, or suppresses the signal. This state layer is the central technical distinction because it provides a deterministic control point between extraction and action.

### 8.5 Bounded Tool Runtime

Each assistant capability is defined by an input contract, output contract, permitted tool list, retrieval strategy, trust policy, and confirmation requirement. The runtime exposes only the permitted tool subset and records the resulting action path.

### 8.6 Action Layer

The action layer converts authorized outputs into tasks, reminders, alarms, calendar proposals, notifications, or monitoring records. Escalation intensity may depend on decision state, time proximity, priority, completion state, and user preference.

### 8.7 Local Fallback Layer

The local fallback layer preserves essential operation when remote services are unavailable. Local notes, local parsing, cached signal summaries, local reminders, and pending action queues can continue to operate with reduced functionality.

---

## 9. Experimental Validation Results

Prototype validation has been performed using a mobile application and backend implementation. The validated portions include:

- separation of private-local and smart-memory content;
- creation and retrieval of scoped memory records;
- structured assistant response handling;
- local document parsing and bounded excerpt use;
- task, reminder, and alarm action paths;
- repository scoping by user and environment;
- encrypted handling of integration credentials;
- local fallback stores; and
- automated tests for core domain and runtime behavior.

The invention is presently assessed at **TRL 5: technology validated in a relevant environment**. Production qualification, larger-scale user validation, and security review remain future work.

---

## 10. Aspects Requiring Protection

The following technical aspects should be protected:

1. hybrid extraction from heterogeneous personal information sources;
2. privacy-tiered separation between private-local data and assistant-eligible memory;
3. finite decision-state routing before action execution;
4. structured assistant output with explanation, citations, and bounded actions;
5. controlled tool execution based on capability, user authorization, environment scope, and trust policy;
6. local-first parsing and fallback operation;
7. escalation-aware conversion of decision states into tasks, reminders, alarms, notifications, or calendar proposals; and
8. the combined system architecture linking extraction, state routing, memory, and tool execution.

---

## 11. Candidate Claim Drafting Anchors

These are technical drafting aids for patent counsel.

### 11.1 Independent Method Claim Anchor

A computer-implemented method comprising:

1. receiving user-specific signals from heterogeneous source classes;
2. normalizing each signal into a structured candidate object;
3. applying a privacy gate that excludes private-local content from cloud memory and model retrieval;
4. extracting evidence using deterministic parsing and machine-learning inference;
5. assigning the extracted evidence to a finite operational decision state;
6. generating structured output containing an explanation, citations, and bounded action candidates;
7. exposing only typed tools authorized by user permission, environment scope, capability definition, and trust policy; and
8. executing, queuing, or presenting for confirmation an action based on the decision state.

### 11.2 Independent System Claim Anchor

A cognitive assistance system comprising:

- an ingestion module;
- a privacy gate;
- a hybrid extraction engine;
- a finite decision-state router;
- a structured response generator;
- a bounded tool runtime;
- a scoped memory store;
- an escalation-aware action layer; and
- a local fallback layer.

### 11.3 Dependent Claim Concepts

Dependent claims may cover:

1. routing uncertain but potentially important signals to `monitor`;
2. selecting escalation intensity based on time proximity;
3. exposing only bounded document excerpts or extracted facts to a model;
4. preventing private-local content from becoming a memory or embedding record;
5. binding records and tools to a user/environment scope;
6. requiring confirmation for external or high-impact side effects;
7. generating relevance explanations from extracted evidence and metadata;
8. promoting accepted signals into action records; and
9. continuing local reminders, parsing, and cached-signal display when remote services are unavailable.

---

## 12. Recommended Filing Annexures

| Annexure | Content |
| :--- | :--- |
| A1 | System architecture diagram. |
| A2 | Decision-state routing diagram. |
| A3 | Data model summary for signal, memory, decision, and action objects. |
| A4 | Prior-art and novelty matrix. |
| A5 | Prototype validation summary. |
| A6 | Security summary for privacy-tiered memory, environment scoping, and encrypted credentials. |

---

## 13. Filing Summary

The invention should be filed as a system, method, and computer-readable-medium disclosure. The core technical contribution is a control architecture that transforms heterogeneous personal data into finite decision states before any assistant action is executed. This architecture is strengthened by hybrid extraction, privacy-tiered memory, structured output, scoped tool access, and trust-policy confirmation.

---

## End of Document
