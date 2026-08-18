# 🍬 Nassau Candy Supply Chain Optimization & ML Dashboard

##  Project Overview

Nassau Candy, a major distributor operating a complex multi-echelon supply chain, was experiencing unpredictable shipping delays. Initial management hypotheses pointed to factory strain and capacity bottlenecks, leading to a proposed multi-million-dollar "load balancing" strategy. 

This project utilizes data engineering and machine learning to analyze the true root causes of these supply chain delays. By cleaning highly corrupted enterprise data (handling 1,300-day time-traveling outliers) and engineering operational features, we successfully tested management's hypothesis and built a predictive decision-support system.

##  Key Insights: The "Dusty Shelf" Paradox

A Random Forest Regression model was trained to predict delivery lead times, achieving a highly accurate **1.55-day Mean Absolute Error (MAE)**. 

The AI simulation revealed a counter-intuitive operational truth known as the **"Dusty Shelf" Paradox**:
* Busy factories with high 30-day backlogs actually process orders **faster**. High-volume "bestseller" items are prioritized and staged efficiently by warehouse staff.
* Rare, low-volume items sitting on the "dusty shelf" take much longer to pick and pack.
* Therefore, attempting to "load balance" the factories to reduce strain would disrupt this natural efficiency and actually *increase* delays.

**Business Recommendation:** The model proved that Carrier Speed (Shipping Tier) is the overwhelming #1 driver of delays. Nassau Candy should abandon factory reallocation and immediately redirect those funds toward renegotiating ground-shipping contracts with enterprise carriers (UPS/FedEx).

##  Tech Stack & Methodology

* **Data Engineering:** `Python`, `Pandas`, `NumPy` (Logarithmic Transformations for wholesale order normalization).
* **Machine Learning:** `Scikit-Learn` (Random Forest Regressor, Feature Importance analysis).
* **Deployment:** `Streamlit` (Interactive web dashboard for executive scenario testing).

##  How to Run the Dashboard Locally

If you want to run this AI dashboard on your own machine:
1. Clone this repository to your local computer.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt