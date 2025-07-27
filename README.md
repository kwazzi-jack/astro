# Astro: AI Assistant for Radio Astronomy & Interferometry

> **Work in Progress:** This project is under active development and subject to change.

## Overview

Astro is a research prototype application for agentic software in radio astronomy and interferometry, leveraging AI and high-level ML agents. The project aims to provide modular, extensible tools for astronomers and researchers, with a focus on agent-based computation and multi-agent frameworks.

## Features

- Modular agent architecture based on the actor model
- Integration with LangChain and LangGraph for agent logic
- Streamlit-based user interface (initial prototype)
- Extensible design for future ML/AI models and workflows
- Designed for research, experimentation, and future expansion

## Installation

> **Note:** Installation methods below require testing and may change as the project develops.

Astro is not yet available on PyPI. You can install the latest development version using one of the following methods:

### 1. Clone & Install (Standard Python)

```bash
git clone https://github.com/kwazzi-jack/astro.git
cd astro
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Direct Pip Install from GitHub

```bash
pip install git+https://github.com/kwazzi-jack/astro.git
```

### 3. Using `uv` (Recommended)

```bash
git clone https://github.com/kwazzi-jack/astro.git
cd astro
uv venv
uv pip install -e .
```

### 4. Using Poetry

```bash
git clone https://github.com/kwazzi-jack/astro.git
cd astro
poetry install
poetry shell
```

### 5. Using Conda

```bash
git clone https://github.com/kwazzi-jack/astro.git
cd astro
conda create -n astro python=3.12
conda activate astro
pip install -e .
```

> **Note:** Astro uses [`uv`](https://github.com/astral-sh/uv) for environment management with `python3.12`. You may also use standard Python virtual environments.

## Usage

Astro currently runs as a Streamlit application:

```bash
streamlit run astro/app.py
```

> **TODO:** Add more usage examples and CLI instructions as the project develops.

## Dependencies

All dependencies are listed in [`pyproject.toml`](./pyproject.toml). Key libraries include:
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Streamlit](https://streamlit.io/)

## Documentation

Project documentation is under development.

> **TODO:** Add links to documentation and API reference.

## Citation

If you use Astro in academic work, please cite this repository.

> **TODO:** Add citation information when available.

## Contributing

Contributions are welcome! Please use GitHub Issues and Pull Requests for bug reports, feature requests, and code contributions.

- For design philosophy and research background, see the [Wiki](https://github.com/<your-username>/astro/wiki).
- For code contributions, see [`CONTRIBUTING.md`](./CONTRIBUTING.md) (to be created).

## License

Astro is open-source. The recommended license is [MIT](https://opensource.org/licenses/MIT), but please ensure the project name "Astro" does not infringe on existing trademarks or copyrights.

> **TODO:** Add LICENSE file and confirm naming.

## Contact & Support

For questions, issues, or support, please use [GitHub Issues](https://github.com/<your-username>/astro/issues) or [Discussions](https://github.com/<your-username>/astro/discussions).

---

## 2. Design Philosophy

### 2.1 What is an Agent?

The idea behind an agent for agentic software is a not precisely-defined, especially from a design approach. That is, what makes a component of your software an "agent". Several definitions have been discussed but there does not appear to be a consensus on how it links to an LLM and how to design one. Moreover, the design approach taken is usually taken is one that meets the needs of the underlying software framework, e.g. Llama-Index, LangChain, etc., or that integrates quickly and easily with existing workflows. Both have their strengths and weaknesses.

For Astro, we will extend the [*actor model*](https://en.wikipedia.org/wiki/Actor_model) to promote the same decoupled and structured approach of the [*model context protocol* (MCP)](https://en.wikipedia.org/wiki/Model_Context_Protocol), but across all interaction types. An actor is an asynchronous entity that can act on a given observation under some context. We can adapt *effects* onto the actor as additional protocols for the actor to implement. Each effect can be introduced as a module to plug into the actor to enhance their capabilities and processing. We define a structured block of code to be *agentic* if at any point during its initialisation or execution, processing or control is deferred to an AI model (not only an LLM). Therefore, we could have *agentic functions* if the function invokes a given AI model for a response, even if that only represents 1% of its makeup. Moreover, we could have *agentic classes* or simply *agents* for the same reason. The key idea is to acknowledge the non-deterministic nature of AI models inside a deterministic labelled code block. Even if the non-deterministic behaviour is not dominant, we would have to label the chain of execution as non-deterministic since one link in that chain is non-deterministic. This is known as the *non-determinism closure* (CITE).

TODO: Add some maths to drive the point home.

Without this non-deterministic or *agentic* component, the computation falls back to its regular deterministic properties. Thus, under our definition, an agent is an *agentic actor* or an actor than has agency.

Another way to understand this is by abstracting the layers of controlled execution when running a given program. To simplify, in non-agentic software, we have the user layer and below it the system layer. The system layer refers to the direct control and execution conducted by the system given the user provided program. The user does not have any control over this process and can only interact with the system layer through input programs or I/O effects. In these scenarios, the control is released and given to the user to perform the next action. Therefore, during the programs execution, we potentially have several motions of releasing control and passing of messages between the layers till the programs termination where the control is given back to the user.

With agents, we can insert an agent layer between the user and the system. Now, the execution and control, when it requires further information or input, can release to the agent layer *before* the user layer. The agent has its own control and processing that it can conduct with the system layer outside of the user layer. This is where the agency and non-determinism appears because some of the control is deferred to the agent. This approach has its benefits, concerns and limitations, but for now explains the general abstraction of how Astro will function with any user.

Another important assumption we make is that we only allow one non-deterministic effect to be contained in any given actor. This is purely to promote decoupling and safety to ensure that reporting and audit processes can be properly managed. If two or more non-deterministic effects are contained in a single agent, both the reporting and understanding of how the agent functions and how it behaves becomes more complex and harder to control and design. Therefore, resource permitting, if an agent requires the use of two non-deterministic effects, i.e. an LLM and a LRM, one should be relegated to a separate agent structure and the original should communicate with this new agent.

We chose the actor model as a foundation not only for the above purpose, but also to allow for more complex multi-agent frameworks to be designed. If we follow a purely deterministic system, encoding such complex systems would grow increasingly complicated to design, build and maintain. It allows the system to be more modular and provide better information during the software's runtime. Moreover, the actor model approach allows for an approach not easily allowed by others: idleness. Since each actor is decoupled, there is a potential for agents to persist in idle-states. This is a form of agent-based computation not well explored due to the complexity of such an approach. Since an agent is an actor, they only need to act when asked. Therefore, from instantiation to cleanup, an agent can switch between active and idle modes. There is a lot of potential to explore with idleness and control processes that have not been explored, especially in Astro.

### 2.2 Designing an Agent

As mentioned in [2.1](#21-what-is-an-agent), an agent is an actor with a non-deterministic effect included. In general, we denote this as the *effect* module or `EffModule`. How this is used within the agent determines the type of agent it becomes. An LLM-based agent would simply implement the effect module as an LLM. For example, an agent with no additional modules would be considered a *reactive agent* whereby the `EffModule` (the LLM) *reacts* to the input provided with *static context*, i.e. system prompt. It does not have access to any other modules to improve its own context and thus enhance the agent output. This leads to our next effect, the *context module* or `ContextModule`.

#### Agent Module Design: Key Points

- **Non-deterministic Effect (NDModule):**
  - Encapsulates any operation or model introducing non-determinism (e.g., LLM, CNN, statistical model, randomness).
  - Extends the definition of agency beyond LLMs to any non-deterministic computation.
  - *Mathematical notation:* If $f$ is a deterministic function and $g$ is a non-deterministic function, then the agentic closure is $h(x) = f(g(x))$ where $g$ introduces non-determinism. [CITE:"non-determinism in computation"]

- **Context Module (ContextModule):**
  - Manages agent-to-environment communication and dynamic context.
  - Implements protocols such as MCP for runtime context acquisition.
  - Distinguishes between *static context* (system prompt, immutable during agent lifetime) and *dynamic context* (mutable, environment-driven).
  - See [Model Context Protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol) [CITE:"model context protocol"].

- **Memory Module (MemoryModule):**
  - Handles the agent’s memory functions, subdivided into:
    - **Working Memory:**
      - Actively used, transient information for immediate reasoning or computation.
      - Limited capacity, frequently updated and overwritten.
      - Analogous to the cognitive concept of working memory [CITE:"working memory cognitive science"].
    - **Short Term Memory:**
      - Temporarily stores recent observations, actions, or events.
      - Larger capacity than working memory, not actively manipulated.
      - Can be recalled or moved to working memory if needed.
    - **Persistent Memory (Long Term):**
      - Stores information across agent lifetimes or sessions.
      - Used for learning, experience, or persistent state.
  - **Distinction:**
    - Working memory is a managed subset of state, used for current computation.
    - Short term memory is for recent history, not actively used.
    - Persistent memory is for long-term storage.

- **State Module (StateModule):**
  - Represents the agent’s current configuration, including all persistent and transient variables.
  - State is broader than working memory; it includes all agent properties, not just those actively used.
  - *Mathematical notation:* The agent’s state at time $t$ can be denoted as $S_t$, with working memory $W_t \subseteq S_t$ and short term memory $STM_t \subseteq S_t$.
  - See [Section 2.1](#21-what-is-an-agent) for foundational discussion.

- **Communication Module (CommunicationModule):**
  - Facilitates agent-to-agent messaging via the actor model.
  - Enables independence, modularity, and idleness in multi-agent systems.
  - Supports message passing, coordination, and distributed computation [CITE:"actor model communication"].

#### Summary Table

| Module                | Scope/Function                                  | Persistence      | Usage Example                  |
|-----------------------|-------------------------------------------------|------------------|-------------------------------|
| EffectModule              | Non-deterministic effect/model                  | Transient        | LLM, CNN, randomness          |
| ContextModule         | Agent-to-Environment interaction/context           | Dynamic/Static   | MCP, system prompt            |
| MemoryModule          | Working, short-term, persistent memory          | Varies           | Recent messages, experience   |
| StateModule           | All agent variables/configuration               | Persistent       | Position, goals, flags        |
| CommunicationModule   | Agent-to-agent messaging                        | Transient        | Actor model message passing   |

#### Additional Notes

- **Working Memory vs Short Term Memory:**
  - Working memory is for immediate computation; short term memory is for recent history.
  - Both are distinct from the agent’s overall state, which includes all properties.
  - For further reading, see [CITE:"working memory vs short term memory"].

- **Extensibility:**
  - This modular approach allows for future expansion (e.g., PlanningModule, PerceptionModule).
  - Modules can be composed as needed for different agent types.

- **Cross-reference:**
  - For the theoretical foundation of agency and non-determinism, see [Section 2.1](#21-what-is-an-agent).
  - For more on the actor model, see [Actor Model](https://en.wikipedia.org/wiki/Actor_model) [CITE:"actor model"].
