# Adaptive DQN-Based Fraud Detection System

An adaptive fraud detection system using **Deep Q-Networks (DQN)**, ensemble disagreement, human review, and human-feedback-driven reinforcement learning.

## Project Overview

This project treats fraud detection as a sequential decision-making problem using Reinforcement Learning. The DQN agent selects one of three actions:

| Action | Description |
|---|---|
| `0` | AUTOMATIC_LEGITIMATE |
| `1` | FLAG_FRAUD |
| `2` | HUMAN_REVIEW |

The system combines automated fraud detection with a human-in-the-loop mechanism. Human feedback can be incorporated into the DQN so that the model can be evaluated before and after feedback-based learning.

## Objectives

1. Detect fraudulent transactions using a DQN agent.
2. Identify uncertain transactions requiring human review.
3. Incorporate ensemble disagreement into the RL state.
4. Collect human decisions for review cases.
5. Use human feedback to update the DQN.
6. Compare the original and feedback-updated models.
7. Analyze classification performance and RL rewards.
8. Generate visual analytics.
9. Provide a React-based frontend for demonstration.

## System Architecture

```text
Transaction Dataset
        |
        v
Data Preprocessing
        |
        v
Ensemble Fraud Detection
        |
        v
Disagreement Data
        |
        v
DQN Environment
        |
        v
DQN Training
        |
        v
Original DQN
        |
        v
DQN Evaluation
        |
        +----------------------+
        |                      |
        v                      v
Automatic Decision       Human Review
                               |
                               v
                        Human Feedback
                               |
                               v
                     Feedback Learning
                               |
                               v
                      Updated DQN
                               |
                               v
                       Final Evaluation
                               |
                               v
                    Before/After Comparison
                               |
                               v
                         Visualization
                               |
                               v
                        React Frontend
```

## Technology Stack

### Machine Learning / RL
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Deep Q-Network
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
- Human-feedback learning

### Visualization
- Matplotlib
- CSV-based analysis

### Frontend
- React
- Vite
- JavaScript
- CSS
- Lucide React

### Hardware Used
- NVIDIA GeForce RTX 3050 Laptop GPU
- CUDA
- PyTorch CUDA build

## Project Structure

```text
MP PWP-1/
│
├── data/
│   └── processed/
│       ├── dqn_evaluation_results.csv
│       ├── dqn_evaluation_summary.csv
│       ├── dqn_human_review_results.csv
│       ├── human_feedback.csv
│       ├── feedback_learning_data.csv
│       ├── feedback_dqn_evaluation_results.csv
│       └── dqn_before_after_comparison.csv
│
├── rl/
│   ├── dqn_agent.py
│   ├── dqn_environment.py
│   ├── replay_buffer.py
│   ├── train_dqn.py
│   ├── evaluate_dqn.py
│   ├── human_feedback.py
│   ├── create_human_feedback.py
│   ├── collect_human_feedback.py
│   ├── simulate_human_feedback.py
│   ├── feedback_learning.py
│   ├── update_dqn_from_feedback.py
│   ├── evaluate_feedback_dqn.py
│   ├── compare_dqn_results.py
│   └── generate_final_graphs.py
│
├── saved_models/
│   ├── dqn_fraud_agent.pt
│   └── dqn_fraud_agent_feedback_updated.pt
│
├── results/
│   └── graphs/
│       ├── 01_classification_performance.png
│       ├── 02_error_comparison.png
│       ├── 03_human_review_reduction.png
│       ├── 04_original_action_distribution.png
│       ├── 05_feedback_action_distribution.png
│       ├── 06_total_reward_comparison.png
│       └── 07_average_reward_comparison.png
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

## DQN Environment

Current environment configuration:

```text
Transactions      : 74,020
Transaction       : 30
RL features       : 14
State size        : 44
Number of actions : 3
```

Actions:

```text
0 -> AUTOMATIC_LEGITIMATE
1 -> FLAG_FRAUD
2 -> HUMAN_REVIEW
```

## Reward Structure

| Situation | Reward |
|---|---:|
| Correct legitimate decision | +2 |
| Correct fraud decision | +5 |
| Human review on fraud | +3 |
| Human review on legitimate | -1 |
| False positive | -4 |
| False negative | -10 |

The reward structure strongly penalizes missed fraud while still accounting for human-review cost.

## DQN Architecture

```text
Input: 44 features
        |
        v
Fully Connected: 256
        |
        v
Fully Connected: 128
        |
        v
Fully Connected: 64
        |
        v
Output: 3 Q-values
        |
        v
Three possible actions
```

Configuration:

```text
State size       : 44
Action size      : 3
Hidden layers    : [256, 128, 64]
Learning rate    : 0.0001
Gamma            : 0.99
Initial epsilon  : 1.0
Minimum epsilon  : 0.05
Epsilon decay    : 0.995
Total parameters : 52,867
```

## Experience Replay

Each experience contains:

```text
state
action
reward
next_state
done
```

Training configuration:

```text
Replay capacity         : 10,000
Batch size              : 64
Target update frequency : 500 steps
```

## Training

Training script:

```text
rl/train_dqn.py
```

Run:

```powershell
python rl/train_dqn.py
```

The trained model is saved to:

```text
saved_models/dqn_fraud_agent.pt
```

Completed training:

```text
Total episodes          : 15
Total environment steps : 1,110,300
Best episode reward     : -48,844
Final epsilon           : approximately 0.9276
```

## Original DQN Evaluation

Run:

```powershell
python rl/evaluate_dqn.py
```

Evaluation dataset:

```text
74,020 samples
```

Original DQN results:

```text
Accuracy  : 0.999811
Precision : 0.999300
Recall    : 1.000000
F1 Score  : 0.999650
ROC-AUC   : 0.999964
PR-AUC    : 0.999771
```

Confusion matrix:

```text
[[54006    14]
 [    0 20000]]
```

Therefore:

```text
True Negatives  : 54,006
False Positives : 14
False Negatives : 0
True Positives  : 20,000
```

## Original Action Distribution

```text
AUTOMATIC_LEGITIMATE : 53,984
FLAG_FRAUD           : 20,001
HUMAN_REVIEW         : 35
```

Human-review percentage:

```text
0.0473%
```

## Human Review

Transactions for which the DQN selects `HUMAN_REVIEW` are placed in the review queue.

Original evaluation:

```text
Human review cases: 35
```

Stored in:

```text
data/processed/dqn_human_review_results.csv
```

The feedback template is:

```text
data/processed/human_feedback.csv
```

Human decisions use:

```text
0 = LEGITIMATE
1 = FRAUD
```

## Human Feedback

Important feedback fields include:

```text
sample_index
dqn_action
dqn_prediction
dqn_fraud_probability
human_decision
feedback_reward
dqn_agrees_with_human
review_status
review_timestamp
ground_truth
```

Reward examples:

```text
Human=0 | Actual=0 | Reward=2
Human=1 | Actual=1 | Reward=5
Human=1 | Actual=0 | Reward=-4
Human=0 | Actual=1 | Reward=-10
```

## Simulating Human Feedback

For development and demonstration:

```powershell
python rl/simulate_human_feedback.py
```

Current completed simulation:

```text
Total reviewed cases    : 35
Human legitimate        : 32
Human fraud             : 3
DQN-human agreements    : 25
DQN-human disagreements : 10
Agreement rate          : 0.7143
Feedback reward total   : 19.0
Average feedback reward : 0.5429
```

Output:

```text
data/processed/human_feedback.csv
```

## Feedback-Based Learning

Run:

```powershell
python rl/update_dqn_from_feedback.py
```

Current experiment:

```text
Feedback experiences : 35
Training steps       : 100
Batch size           : 32
```

Loss changed from:

```text
245.713440
```

to:

```text
27.777891
```

Updated model:

```text
saved_models/dqn_fraud_agent_feedback_updated.pt
```

## Feedback-Updated Evaluation

Run:

```powershell
python rl/evaluate_feedback_dqn.py
```

Output:

```text
data/processed/feedback_dqn_evaluation_results.csv
```

## Before vs After Comparison

Run:

```powershell
python rl/compare_dqn_results.py
```

Output:

```text
data/processed/dqn_before_after_comparison.csv
```

Current results:

| Metric | Original | Feedback Updated |
|---|---:|---:|
| Accuracy | 0.999811 | 0.999297 |
| Precision | 0.999300 | 0.998351 |
| Recall | 1.000000 | 0.999050 |
| F1 Score | 0.999650 | 0.998700 |
| False Positives | 14 | 33 |
| False Negatives | 0 | 19 |
| Human Review Cases | 35 | 13 |
| Human Review % | 0.0473% | 0.0176% |
| Total Reward | 207,914 | 207,596 |

The current feedback experiment reduced human-review cases from 35 to 13, a reduction of 22 cases.

The classification metrics also decreased slightly in this experiment. This result should be reported honestly; human feedback demonstrates an adaptive learning mechanism but does not necessarily improve every metric after a single feedback-learning round.

## Final Graphs

Generate all graphs with:

```powershell
python rl/generate_final_graphs.py
```

Generated files:

```text
results/graphs/
├── 01_classification_performance.png
├── 02_error_comparison.png
├── 03_human_review_reduction.png
├── 04_original_action_distribution.png
├── 05_feedback_action_distribution.png
├── 06_total_reward_comparison.png
└── 07_average_reward_comparison.png
```

## Important CSV Outputs

| File | Current shape |
|---|---:|
| `dqn_evaluation_results.csv` | 74,020 × 11 |
| `dqn_human_review_results.csv` | 35 × 11 |
| `human_feedback.csv` | 35 × 14 |
| `feedback_dqn_evaluation_results.csv` | 74,020 × 11 |
| `dqn_before_after_comparison.csv` | 13 × 4 |

## Saved Models

Original:

```text
saved_models/dqn_fraud_agent.pt
```

Feedback-updated:

```text
saved_models/dqn_fraud_agent_feedback_updated.pt
```

## React Frontend

The project includes a React frontend with the following planned structure:

```text
frontend/
│
├── src/
│   ├── components/
│   │   ├── Sidebar.jsx
│   │   ├── Topbar.jsx
│   │   ├── MetricCard.jsx
│   │   ├── FraudGauge.jsx
│   │   ├── DecisionCard.jsx
│   │   ├── TransactionTable.jsx
│   │   ├── ReviewQueue.jsx
│   │   ├── QValueChart.jsx
│   │   └── ModelComparison.jsx
│   │
│   ├── pages/
│   │   ├── Dashboard.jsx
│   │   ├── TransactionAnalyzer.jsx
│   │   ├── ReviewCenter.jsx
│   │   ├── Analytics.jsx
│   │   └── ModelComparison.jsx
│   │
│   ├── services/
│   │   └── api.js
│   │
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
│
├── package.json
└── vite.config.js
```

Planned frontend features:

- Dashboard
- Transaction Analyzer
- Fraud probability gauge
- DQN decision display
- Human Review Center
- Analytics
- Model comparison
- Reward analysis
- Action distribution
- Classification metrics
- Human-review statistics
- Adaptive-learning demonstration

## Frontend Installation

From the `frontend` directory:

```powershell
npm install
npm install lucide-react
npm run dev
```

## Demonstration Workflow

The final demonstration should follow:

```text
1. Open Dashboard
        |
        v
2. View fraud statistics
        |
        v
3. Analyze a transaction
        |
        v
4. Show DQN probability and decision
        |
        v
5. Open Human Review Center
        |
        v
6. Provide human decision
        |
        v
7. Show feedback reward
        |
        v
8. Demonstrate feedback learning
        |
        v
9. Compare Original vs Updated DQN
        |
        v
10. Display graphs and metrics
```

## Complete ML Pipeline

```powershell
python rl/train_dqn.py
python rl/evaluate_dqn.py
python rl/create_human_feedback.py
python rl/simulate_human_feedback.py
python rl/update_dqn_from_feedback.py
python rl/evaluate_feedback_dqn.py
python rl/compare_dqn_results.py
python rl/generate_final_graphs.py
```

## Project Status

```text
Phase 4A - DQN Training                    COMPLETED
Phase 4B - DQN Evaluation                  COMPLETED
Phase 4C - Human Feedback                  COMPLETED
Phase 4D - Feedback Learning               COMPLETED
Phase 4E - Feedback DQN Evaluation         COMPLETED
Phase 4F - Before/After Comparison         COMPLETED
Final Visualization                        COMPLETED
React Frontend                             IN PROGRESS
Backend/API Integration                    NEXT
Final Demonstration                        NEXT
```

## Future Improvements

Potential improvements include:

- More human feedback samples
- Multiple feedback-learning rounds
- Better feedback weighting
- More balanced feedback sampling
- Conservative policy updates
- Combining original replay experiences with human feedback
- Improved reward tuning
- More extensive validation
- Live transaction inference through an API
- Authentication and role-based access
- Persistent human-review workflow
- Real-time dashboard updates

## Conclusion

This project demonstrates the integration of:

```text
Machine Learning
        +
Ensemble Fraud Detection
        +
Reinforcement Learning
        +
Deep Q-Network
        +
Human-in-the-Loop
        +
Human Feedback
        +
Adaptive Model Updating
        +
Visualization
        +
React Frontend
```

The final system demonstrates how a fraud detection agent can make automated decisions while retaining a human-review mechanism and using human feedback as an additional learning signal.

## Contributors
 - Salike Akshat - salikeakshat28@gmail.com
 - Rohit Chennam - rohitchennam1@gmail.com
