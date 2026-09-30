Autonomous Research & Decision Agent

A modular agentic AI system that transforms a user goal into a sequence of planning, tool selection, execution, observation, and decision steps.

The project is designed to demonstrate practical agentic AI engineering rather than only LLM generation. It includes a deterministic baseline agent, modular tool architecture, state and execution management, evaluation infrastructure, a learned tool-selector candidate, FastAPI backend, and React frontend.

📌 Overview

The Autonomous Research & Decision Agent accepts a natural-language goal and determines how to solve it using available tools.

The current Phase 6 execution flow is:

User Goal
   ↓
Task Creation
   ↓
Planning
   ↓
Tool Selection
   ↓
Tool Execution
   ↓
Observation
   ↓
Decision
   ↓
Final Answer

For example:

Goal:
30*7/2+157-254

        ↓

Tool Selection:
calculate

        ↓

Tool Execution:
30*7/2+157-254

        ↓

Result:
8

        ↓

Final:
8.0

The project currently uses a deterministic baseline controller as the production execution path. A TF-IDF + Logistic Regression tool-selector model has also been trained and evaluated as a candidate, but it is intentionally not yet used as the production controller.

🎯 Project Objectives

The main objectives are:

Build a modular agent architecture.
Separate planning, decision-making and tool execution.
Maintain complete execution state.
Support multiple tools through a registry.
Handle execution failures and bounded retries.
Prevent repeated tool-call loops.
Evaluate agent trajectories systematically.
Train a lightweight learned tool selector.
Expose the agent through a FastAPI API.
Provide a React-based frontend.
Keep the architecture extensible for future LLM-based planning and tool calling.
🧠 Architecture
                         ┌──────────────────────┐
                         │       User           │
                         │       Goal           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     FastAPI API      │
                         │  /api/agent/run      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Agent Service     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Agent Task        │
                         │  task_id + goal      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Planner         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Agent Controller     │
                         │                      │
                         │ Tool Selection       │
                         │ Confidence           │
                         │ Loop Protection      │
                         │ Step Limits          │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         │                      │
                         ▼                      ▼
                ┌────────────────┐     ┌──────────────────┐
                │   Calculator   │     │ Knowledge Search │
                └───────┬────────┘     └────────┬─────────┘
                        │                       │
                        └──────────┬────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │    Tool Executor     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Observation      │
                         │   Tool Result/Error  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Agent State      │
                         │                      │
                         │ Tool Calls           │
                         │ Results              │
                         │ Observations         │
                         │ Decisions             │
                         │ Errors                │
                         │ Retries               │
                         │ Replans               │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Final Decision     │
                         │   / Final Answer     │
                         └──────────────────────┘
🔗 Relationship With EKI-RAG

This project is the second stage of a broader AI engineering progression.

Project 1 — EKI-RAG
Knowledge Retrieval
        ↓
Hybrid Search
        ↓
Multi-Hop Retrieval
        ↓
Grounded Generation
Project 2 — Autonomous Research & Decision Agent
Goal
 ↓
Planning
 ↓
Tool Selection
 ↓
Tool Execution
 ↓
Observation
 ↓
Decision

The architecture is designed so that EKI-RAG can become one of the agent's knowledge-retrieval tools.

This creates a progression from:

AI Knowledge System
        ↓
AI Agent
        ↓
Agent + Knowledge Retrieval
🛠️ Current Tools

The current Phase 6 production execution path contains:

Calculator

Tool name:

calculate

Used for arithmetic expressions such as:

30*7/2+157-254

The controller recognizes both explicit calculation requests:

Calculate 25 * 4

and standalone arithmetic expressions:

25 * 4
Knowledge Search

Tool name:

search_knowledge

Used for research-oriented requests.

Examples:

Tell me about Kubernetes.
Explain Transformer architecture.
Find information about RAG.
🧩 Core Components
Agent Task

Represents the task being solved.

Contains information such as:

task_id
goal

Task IDs are generated by the backend when execution begins.

Agent State

Maintains the complete execution state.

The state tracks:

task
plan
current step
tool calls
tool results
observations
decisions
errors
retry count
replan count
metadata

This makes the agent trajectory observable and testable.

Planner

The current project contains a deterministic baseline planner.

Its responsibility is to create the initial execution plan before the agent begins execution.

Controller

The controller is responsible for:

selecting tools
respecting planned tools
detecting unsupported tasks
enforcing confidence thresholds
enforcing maximum steps
detecting repeated tool calls
deciding whether to execute or finish
generating the final answer
Tool Registry

The ToolRegistry provides a central mechanism for registering and discovering tools.

This allows additional tools to be added without rewriting the complete agent architecture.

Tool Executor

The executor is responsible for actually executing the selected tool.

The controller decides:

What should be executed?

The executor handles:

Execute it.

This separation keeps decision-making and execution responsibilities independent.

🔄 Agent Execution Example

Input:

30*7/2+157-254

The system performs:

1. Create AgentTask
        ↓
2. Generate task_id
        ↓
3. Create baseline plan
        ↓
4. Controller analyzes goal
        ↓
5. Detect standalone arithmetic expression
        ↓
6. Select calculate
        ↓
7. Create ToolCall
        ↓
8. ToolExecutor executes calculator
        ↓
9. Store ToolResult
        ↓
10. Create observation
        ↓
11. Controller finishes
        ↓
12. Return final answer

Result:

8.0
🧠 Learned Tool Selector

A separate learned tool-selection experiment was developed using:

TF-IDF
   +
Logistic Regression

The classifier predicts:

calculate
search_knowledge
none

The purpose is to investigate whether a learned model can replace or augment deterministic tool selection.

📊 Tool Selector Dataset

The final training dataset contains:

11,008 tasks

with separate:

Training
Validation
Test

splits.

The dataset contains:

calculation tasks
research tasks
unsupported tasks
boundary cases

The dataset was checked for:

exact duplicate goals
task-ID leakage
cross-split leakage
class distribution

The official Phase 5 benchmark is kept separate from the generated training/test dataset.

📈 Tool Selector Results

The learned selector achieved:

Validation Accuracy: 99.73%

Test Accuracy:       99.73%

Test Macro F1:       99.74%

Group-aware CV:
Accuracy ≈ 99.93%

However, these results should not be interpreted as equivalent to real-world performance.

The dataset is synthetically generated and relatively structured, so lexical patterns can make the classification problem easier.

Therefore:

Model Status:
Candidate

rather than:

Production Model

The deterministic controller remains the current Phase 6 production execution path.

⚠️ Model Evaluation Philosophy

The project deliberately avoids treating one accuracy number as proof of model quality.

Evaluation considers:

tool-selection accuracy
tool-argument accuracy
execution success
task completion
recovery
unnecessary tool calls
repeated tool calls
trajectory accuracy
final-answer accuracy
confidence
leakage
class balance

For the learned selector, additional evaluation against the official held-out benchmark is required before replacing the deterministic controller.

🧪 Evaluation Framework

The project contains a dedicated evaluation module.

Metrics include:

Exact Match
Tool Selection Accuracy
Tool Argument Accuracy
Execution Success Rate
Task Completion Rate
Recovery Success Rate
Unnecessary Tool Call Rate
Repeated Tool Call Rate
Trajectory Accuracy
Final Answer Accuracy

The goal is to evaluate the agent trajectory, not just its final answer.

🛡️ Reliability Features

The agent contains several safety and reliability mechanisms.

Maximum Steps

Prevents unlimited execution.

max_steps = 8
Retry Limits

Tool failures can be retried within configured limits.

Replan Limits

Replanning is bounded to prevent uncontrolled loops.

Repeated Tool Detection

Identical tool calls are detected and limited.

Confidence Thresholds

The controller can stop when tool-selection confidence is below the configured threshold.

Terminal State Detection

The agent recognizes terminal states such as:

COMPLETED
FAILED
ABORTED
🌐 FastAPI Backend

The backend exposes the agent through FastAPI.

Main endpoints:

GET /
GET /api/health
POST /api/agent/run
GET /api/tools

Interactive API documentation is available through FastAPI's generated documentation when the backend is running.

📡 Example API Request
POST /api/agent/run
Content-Type: application/json

Request:

{
  "goal": "30*7/2+157-254",
  "max_steps": 8
}

Example response structure:

{
  "task_id": "task-xxxxxxxx",
  "goal": "30*7/2+157-254",
  "status": "completed",
  "final_answer": "8.0",
  "tool_calls": [
    {
      "tool_name": "calculate",
      "arguments": {
        "expression": "30*7/2+157-254"
      }
    }
  ]
}
🖥️ React Frontend

The Phase 6 frontend provides a simple interface for:

entering a goal
configuring maximum steps
executing the agent
viewing execution status
viewing tool calls
viewing tool results
viewing decision traces
viewing the final answer
monitoring API health

Current frontend stack:

React
Vite
CSS
📁 Project Structure
Autonomous Research Decision Agent/
│
├── app/
│   ├── __init__.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent_loop.py
│   │   ├── controller.py
│   │   ├── planner.py
│   │   ├── schemas.py
│   │   └── state.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── agent_service.py
│   │
│   └── tools/
│       ├── __init__.py
│       ├── base.py
│       ├── calculator.py
│       ├── data_analysis.py
│       ├── executor.py
│       ├── knowledge_search.py
│       ├── registry.py
│       ├── report_generator.py
│       └── summarizer.py
│
├── configs/
│   ├── agent.yaml
│   ├── evaluation.yaml
│   └── tools.yaml
│
├── data/
│   ├── evaluation/
│   └── tasks/
│       └── agent_evaluation.jsonl
│
├── experiments/
│   └── README.md
│
├── frontend/
│   ├── public/
│   │   └── favicon.svg
│   │
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   └── styles/
│   │       └── global.css
│   │
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   └── tool_selector/
│       └── tool_selector_tfidf_logistic.pkl
│
├── scripts/
│   ├── __init__.py
│   ├── evaluate_agent.py
│   ├── generate_dataset.py
│   └── run_agent.py
│
├── tests/
│   ├── __init__.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── test_agent_loop.py
│   │   ├── test_controller.py
│   │   ├── test_recovery.py
│   │   └── test_stopping.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── test_evaluator.py
│   │   └── test_metrics.py
│   │
│   └── tools/
│       ├── __init__.py
│       ├── test_calculator.py
│       ├── test_executor.py
│       └── test_registry.py
│
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   ├── nginx.conf
│   └── docker-compose.yml
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

Generated datasets, model artifacts, caches and large experiment outputs should generally remain outside Git unless there is a deliberate reason to version them.

⚙️ Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Autonomous Research Decision Agent"
2. Create the Conda environment
conda create -n agentic-ai python=3.11.16

Activate it:

conda activate agentic-ai
3. Install Python dependencies
pip install -r requirements.txt
🚀 Run the Backend

From the project root:

uvicorn app.main:app --reload

The backend will run at:

http://127.0.0.1:8000

Health endpoint:

http://127.0.0.1:8000/api/health

FastAPI documentation:

http://127.0.0.1:8000/docs
🖥️ Run the Frontend

Open another terminal:

cd frontend

Install dependencies:

npm install

Run development server:

npm run dev

The frontend will normally be available at:

http://127.0.0.1:5173
🧪 Run Tests

From the project root:

pytest -q

Run agent tests:

pytest tests/agent -q

Run tool tests:

pytest tests/tools -q

Run evaluation tests:

pytest tests/evaluation -q
🧮 Run the Agent From CLI

Example:

python scripts/run_agent.py "Calculate 25 * 4"

Example:

python scripts/run_agent.py "30*7/2+157-254"

Example research request:

python scripts/run_agent.py "Tell me about Kubernetes"
📊 Generate Evaluation Dataset

The project includes a dataset-generation pipeline for tool-selector experimentation.

Example:

python scripts/generate_dataset.py

The generated dataset should be treated as training/evaluation infrastructure, not as proof of real-world model performance.

🔬 Evaluate the Agent

Run:

python scripts/evaluate_agent.py

The evaluation pipeline reports metrics covering tool selection, execution, recovery, trajectory and final-answer behavior.

🧱 Design Principles

The project follows several engineering principles.

1. Separate Decision From Execution

The controller decides:

What should happen?

The executor handles:

Actually execute it.
2. Keep State Explicit

Important execution information is stored in AgentState instead of being hidden inside individual functions.

3. Establish Baselines Before Adding ML

The deterministic controller provides a reference implementation before introducing a learned selector.

4. Evaluate Before Promoting Models

A trained model is not automatically considered production-ready because its test accuracy is high.

5. Keep Failure Modes Observable

The system records:

tool calls
results
observations
decisions
errors
retries
replans

This makes debugging and evaluation easier.

🔐 Security Considerations

The current project is a development/demo architecture and should not be treated as an unrestricted production agent.

For production deployment, additional controls should be implemented around:

authentication
authorization
tool permissions
input validation
rate limiting
secrets management
network access
filesystem access
external API access
execution timeouts
audit logging
human approval for high-impact actions

Particularly, tools that can modify external systems should require stronger controls than read-only tools.

📦 Current Phase Status
Phase 1 — Agent Foundation

Status: Complete

Implemented:

agent task/state foundation
schemas
basic agent architecture
tests
Phase 2 — Tool System

Status: Complete

Implemented:

Tool abstraction
Tool Registry
Tool Executor
Calculator
Knowledge Search
Tool tests
Phase 3 — Agent Loop

Status: Complete

Implemented:

controller
agent loop
deterministic tool selection
CLI execution
execution tracing
Phase 4 — Reliability

Status: Complete

Implemented:

retry handling
bounded recovery
stopping conditions
repeated-call detection
execution limits
configuration
Phase 5 — Evaluation & Learned Tool Selector

Status: Complete as a candidate pipeline

Implemented:

evaluation framework
benchmark dataset
synthetic training dataset
leakage checks
group-aware splitting
TF-IDF + Logistic Regression tool selector
model evaluation
experiment reporting

The learned selector remains a candidate model, not the production controller.

Phase 6 — Application Layer

Status: Complete

Implemented and verified:

FastAPI backend
React frontend
frontend ↔ backend integration
task ID generation
calculator execution
raw arithmetic detection
tool execution results
decision trace
API health monitoring

Verified example:

Input:
30*7/2+157-254

Tool:
calculate

Result:
8

Final Answer:
8.0

Status:
completed
🔮 Future Work

Potential future development includes:

Learned Tool Selector

Evaluate the trained selector against the official held-out benchmark and harder natural-language inputs before production integration.

LLM-Based Planning

Replace or augment deterministic planning with an LLM-based planner.

More Tools

Potential tools include:

RAG Search
Database Query
Data Analysis
Web Search
Report Generation
Summarization
File Analysis
Genuine Replanning

Move from bounded retry/replan behavior toward genuine alternative-plan generation.

Persistent Agent State

Store long-running task state in a persistent database.

Asynchronous Execution

Support long-running research tasks through background workers.

Observability

Add:

structured logs
metrics
traces
execution dashboards
latency monitoring
tool-level failure monitoring
Production Security

Add:

authentication
RBAC
tool-level permissions
rate limiting
secrets management
approval workflows
📚 Related Project
EKI-RAG — Enterprise Knowledge Intelligence RAG

The first project in this progression focuses on:

Document Ingestion
        ↓
Chunking
        ↓
Embeddings
        ↓
Dense Retrieval
        ↓
BM25
        ↓
Hybrid Retrieval
        ↓
Multi-Hop Retrieval
        ↓
LLM Generation
        ↓
Grounded Answer

The Autonomous Research & Decision Agent extends the concept from:

"Retrieve knowledge and answer"

toward:

"Determine what needs to be done,
choose the appropriate capability,
execute it,
observe the result,
and decide what to do next."
👨‍💻 Author

Harshal Badgujar

AI/ML Engineer

GitHub:
https://github.com/BadgujarHarshal

LinkedIn:
https://linkedin.com/in/badgujarharshal

Email:
harshalbadgujar987@gmail.com

📄 License

This project is intended as a personal AI/ML engineering project and portfolio project.

Add a specific open-source license such as MIT only if you intentionally want to release the repository under that license.

⭐ Project Summary
Autonomous Research & Decision Agent
│
├── Agent Architecture
│   ├── Planner
│   ├── Controller
│   ├── State Manager
│   ├── Tool Registry
│   └── Tool Executor
│
├── Tools
│   ├── Calculator
│   └── Knowledge Search
│
├── ML
│   └── TF-IDF + Logistic Regression
│       └── Learned Tool Selector Candidate
│
├── Evaluation
│   ├── Tool Selection
│   ├── Execution
│   ├── Recovery
│   ├── Trajectory
│   └── Final Answer
│
├── Backend
│   └── FastAPI
│
├── Frontend
│   └── React + Vite
│
└── Engineering
    ├── Tests
    ├── Error Handling
    ├── Retry Limits
    ├── Loop Protection
    └── Execution Tracing

Current verified Phase 6 capability: a user can submit a goal through the React UI, the FastAPI backend creates the agent task, the deterministic controller selects an appropriate tool, the tool executes, the result is recorded, and the final answer plus execution trace is returned to the frontend.