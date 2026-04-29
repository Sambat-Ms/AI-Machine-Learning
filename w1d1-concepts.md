# Week 1 Day 1 - Homework Assignment

## Mapping the ML Universe

---

## Part 1: Conceptual Exercises

Complete these three exercises before moving on to the coding section. They do not require any code — write your answers as comments or in a separate markdown/text file called `w1d1-concepts.md`.

---

### Exercise A: Classify Real-World ML Problems

For each scenario below, determine:

- Problem type (`classification`, `regression`, `clustering`, or `generative`)
- At least 2 likely input features
- What the model output would be


| Scenario                                            | Problem Type            | Input Features                                                                                                                                                                                                                                                                                     | Output                                                                                         |
| --------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Predict whether a loan applicant will default       | classification          | - monthly/annual income- credit score - payment history - loan amount - loan term - Dept-to-income(DTI) ratio- saving balance - expense-to-income ratio - transaction patterns - bank account balance                                                                                             | - 1 ->Default - 0 -> No default                                                               |
| Forecast next month's energy consumption for a city | Regression              | - past monthly consumption (last month, same month last year ) - Temperature - Season(summer, rainy, winter) - Public holiday - Populationn size - Urban vs rural ratio - GDP or economic activity level - Industrial activity index - Number of households - electricity access rate              | continuous number of energy consumption value                                                  |
| Group customers by purchasing behavior              | - `clustering`          | - Age - Gender - Location - Purchase frequency - Recency - Total spend - Average order value - Number of transactions - Seasonal buying patterns - payment method                                                                                                                                  | - Group 1 - Group 2 - Group 3 - Group 4                                                        |
| Detect fraudulent credit card transactions          | classification problems | - Transaction amount - Average transaction amount - Usual spendig locations - typical transaction time(day/night) - Merchant type(MCC code) - Transaction channel (POS, ATM, Online ) - Currency - Number of transaction s in last 5 min / 1 hour / 24 hours - Total amount spent in last 24 hours | - 1-> fraud - 0-> non-fraud                                                                    |
| Generate product descriptions from an image         | `generative`            | - Object type (shirt, shoes, phone) - Shape and struture - Color - Size or dimension - style (Casual, formal, soprty) - Brand logo                                                                                                                                                                 | **text descriptions**                                                                          |
| Predict the severity (1–5) of a patient's condition | classification problems | - Oxygen saturation (SpO₂) - Respiratory rate - Heart rate - Level of consciousness - Key lab marker (e.g., WBC or lactate)                                                                                                                                                                        | Severity score (1-5):- 1 -> mild - 2 -> Moderate - 3 -> Serious - 4 -> Severe - 5 -> Critical |


**How to think about it**:

- Start by asking: *"What does the model output?"* — is it a category, a number, a group, or new content?
- For input features, ask: *"What information would a human expert look at to make this decision?"*
- Watch out for tricky cases: an output that looks like a number (e.g., a severity score 1–5) might still be treated as classification — think about whether the *distance* between values matters.
- If no labeled examples exist in the scenario, clustering is likely the right framing.
- If the model is asked to *produce* something new rather than predict a pre-existing value, lean toward generative.

---

### Exercise B: Map the ML Lifecycle

**Scenario**: A hospital wants to predict which patients admitted to the ER are at high risk of readmission within 30 days.

For each lifecycle step, write 2–3 sentences describing what would happen in this specific scenario.

```
1. Problem Definition: This is a binary classification problem (readmitted vs not readmitted). The goal is to identify high-risk patients early so the hospital can reduce readmissions and improve patient care.

2. Data Collection:  The hospital gathers historical patient data from electronic health records (EHR), including demographics, diagnoses, lab results, medications, and past admissions. Data on previous readmissions within 30 days is used as the target variable. 

3. EDA & Preprocessing: Data is explored to identify patterns, missing values, and outliers (e.g., unusual lab values or ages). Features are cleaned, encoded (e.g., categorical variables like diagnosis), and normalized if needed.

4. Model Training:Machine learning models (e.g., Logistic Regression, Random Forest, or Gradient Boosting) are trained using the prepared dataset. The model learns patterns that distinguish patients who were readmitted from those who were not. Training is done using a split of the data (e.g., train/test or cross-validation).      
 
5. Evaluation: The model is evaluated using metrics like accuracy, precision, recall, and especially recall or AUC, since missing high-risk patients is costly. Confusion matrix analysis helps understand false positives and false negatives.  
        
6. Deployment: The trained model is integrated into the hospital system (e.g., EHR dashboard). When a patient is admitted or discharged, the model generates a risk score for readmission in real time. Healthcare staff can use this information to plan interventions such as follow-up care or patient education. 
       
7. Monitoring: The model’s performance is continuously tracked to ensure accuracy over time. Data drift (e.g., changes in patient population or treatment practices) is monitored. The model is periodically retrained with new data to maintain reliability and effectiveness.        
```

**How to think about each step**:

- **Problem Definition**: Pin down *what exactly* is being predicted, who will act on the prediction, and what "good enough" looks like in business terms.
- **Data Collection**: What records does the hospital already store? How far back should you go? What's a realistic minimum dataset size?
- **EDA & Preprocessing**: What could be messy or missing in medical records? Are there obvious class imbalances? How do you turn diagnosis codes into model-friendly features?
- **Model Training**: Should you start simple or complex? Why might interpretability matter here more than in other industries?
- **Evaluation**: Why might raw accuracy be misleading? What metric would better capture real usefulness?
- **Deployment**: Who sees the output? What context would a nurse or case manager need alongside a risk score?
- **Monitoring**: What real-world changes could make the model go stale? How would you detect this before users notice?

---

### Exercise C: AI vs ML vs Deep Learning Sorting

Label each system as **Rule-based AI**, **Classical ML**, or **Deep Learning**. Write one sentence justifying your choice.

**1. A chess engine that evaluates positions using a hand-crafted evaluation function**
- Rule-based AI : because the chess engine relies on hand-crafted evaluation rules designed by humans (e.g., piece values, positional rules) rather than learning from data.

**2. A spam filter trained on 10 million labeled emails using random forest**
- Classical ML : because it uses a Random Forest model trained on labeled email data to learn patterns and classify spam. 

**3. GPT-4 generating text responses**
- Rule-based Deep Learning: use large neural networks trained on massive text data to generate human-like responses

**4. Netflix's collaborative filtering recommendation engine**
- Classical ML: because Netflix’s collaborative filtering recommends content by learning patterns from user–item interactions (e.g., similar users or viewing histories)

**5. A thermostat that adjusts temperature based on time-of-day rules**
- Rule-based AI: because it follows predefined time-of-day rules (e.g., increase temperature in the morning, decrease at night) without learning from data.

6. **A CNN that detects tumors in medical images**
- because a Convolutional Neural Network (CNN) automatically learns complex visual features from medical images to detect tumors 

7. **A decision tree that approves/rejects credit card applications based on features**
- Classical ML: because a decision tree learns decision rules from historical data using input features (e.g., income, credit score) to approve or reject applications. 

**How to think about it**:

- Ask: *"Does this system learn from data, or does a human write all the rules?"* — if a human encodes the logic explicitly, it's rule-based AI.
- Ask: *"Is this working with raw unstructured data (images, text, audio) at scale?"* — if yes, Deep Learning is likely involved.
- A decision tree *algorithm* trained on data is Classical ML even though the output looks like a set of rules.
- For ambiguous cases (e.g., collaborative filtering): think about whether the underlying method is a neural network or a mathematical decomposition.

---

## 📤 Submission

1. Save your answers in a file named `w1d1-concepts.md`
2. Add it to your repository at `modules/1-foundations/codes/w1d1/w1d1-concepts.md`
3. Submit the GitHub link via the class submission form

---

