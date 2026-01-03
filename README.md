# Machine Learning from Scratch (Java)
![Java](https://img.shields.io/badge/Language-Java-ED8B00?style=for-the-badge&logo=java&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-From%20Scratch-blue?style=for-the-badge)
![Algorithms](https://img.shields.io/badge/Algorithms-KNN%20%7C%20K--Means%20%7C%20Regression-lightgrey?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

A custom implementation of core Machine Learning algorithms built entirely in **Java** without external ML libraries.

### 🎯 Goal
To deconstruct the "black box" of libraries like Scikit-learn and understand the mathematical foundations of algorithms by engineering them from the ground up.

### ⚡ Algorithms Implemented
* **Linear Regression:** Using Gradient Descent optimization.
* **K-Nearest Neighbors (Classification & Regression):** Custom distance metric calculations.



### 🛠️ Tech Stack
* **Language:** Java (OOP principles)
* **Math:** Linear Algebra & Calculus logic implemented raw.
---

### 🧠 What I Learned
Building this from scratch taught me the "magic" behind libraries like Scikit-Learn:
* **Math to Code:** How to translate mathematical formulas (like Gradient Descent and Euclidean Distance) into efficient Java code.
* **OOP Design:** structuring a clean architecture where different models (KNN, Linear Regression) share a common interface.
* **Data Handling:** Writing manual logic to normalize and split data without relying on Pandas.

### 🚀 Future Improvements
Ways I plan to make this engine better:
* **Add More Algorithms:** Implement Logistic Regression and a simple Neural Network.
* **CSV Support:** Build a parser to read real datasets from `.csv` files automatically.
* **Optimization:** Improve matrix multiplication performance for larger datasets.

### 💻 How to Run
This project is built with standard Java (no external ML dependencies).

**Option 1: Using Eclipse (Recommended)**
1.  Clone this repo: `git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git`
2.  Open **Eclipse** and go to `File > Open Projects from File System`.
3.  Select the project folder.
4.  Navigate to `src/ml/app/Main.java`.
5.  Right-click and select **Run As > Java Application**.

**Option 2: Terminal**
```bash
cd src
javac ml/app/Main.java
java ml.app.Main
