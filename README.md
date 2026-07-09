# 🏋️ Personal Fitness Agent

An AI-powered personal fitness assistant that generates personalized **7-day workout and diet plans** based on each user's profile, fitness goals, and preferences.

Built with **LangGraph**, the system uses a multi-agent architecture where specialized agents collaborate to create a complete weekly fitness plan that can be refined and customized over time.

---

## Features

- AI-powered personalized fitness planning
- Generates a complete 7-day diet plan
- Creates a personalized 7-day workout routine
- Plans can be regenerated and customized according to user preferences
- Uses the current date to generate relevant weekly schedules
- Integrates Google Search for up-to-date information when needed
- Multi-agent architecture powered by LangGraph

---

## Architecture

The application follows a supervisor-based multi-agent architecture.

### Supervisor Agent

The Supervisor Agent acts as the central coordinator. It gathers user information, delegates tasks to the specialized agents, and combines their outputs into a unified weekly fitness plan.

### Workout Agent

Responsible for creating personalized workout routines based on factors such as:

- Fitness goals
- Experience level
- Available equipment
- Workout preferences
- Schedule

### Diet Agent

Generates a personalized nutrition plan by considering:

- Fitness goals
- Dietary preferences
- Allergies or restrictions
- Lifestyle
- Calorie requirements

---

## Tech Stack

- Python
- LangGraph
- LangChain
- Google Search Tool
- Date/Time Tools

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hang-kulung/Personal-Fitness-Agent
cd Personal-Fitness-Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file and add the required API keys.

Example:

```env
GOOGLE_API_KEY=your_api_key
```

### 6. Run the application

From the root directory run:

```bash
python supervisor/main.py
```

---

##  Project Structure

```text
Personal-Fitness-Agent/
│
├── supervisor/
│   └── main.py
│
├── workout_agent/
│
├── diet_agent/
│
├── requirements.txt
├── .env
└── README.md
```

