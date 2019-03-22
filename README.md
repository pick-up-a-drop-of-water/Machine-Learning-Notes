# :book:some of most important concepts in Machine Learning

**1. How would you define Machine Learning?**

> Machine Learning is about building systems that can learn from data. Learning means getting better at some task, given some performance measure.
>

**2. Can you name four types of problems where it shines?**

> Machine Learning is great for complex problems for which we have no algorithmic solution, to replace long lists of hand-tuned rules, to build systems that adapt to fluctuating environments, and finally to help humans learn (e.g., data mining).
>

**3. What is a labeled trained set?**

> A labeled training set is a training set that contains the desired solution (a.k.a. a label) for each instance.
>

**4. What are the two most common supervised tasks?**

> The two most common supervised tasks are regression and classification.
>

**5. Can you name four common unsupervised tasks?**

> Common unsupervised tasks include clustering, visualization, dimensionality reduction, and association rule learning.
>

**6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?**

> Reinforcement Learning is likely to perform best if we want a robot to learn to walk in various unknown terrains since this is typically the type of problem that Reinforcement Learning tackles. It might be possible to express the problem as a supervised or semi-supervised learning problem, but it would be less natural.

**7. What type of algorithm would you use to segment your customers into multiple groups?**

> If you't know how to define the groups, then you can use a clustering algorithm (unsupervised learning) to segment your customers into clusters of similar customers. However, if you know what groups you would like to have, then you can feed many examples of each group to classification algorithms (supervised learning), and it will classify all your customers into these groups.
>

**8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?**

>Spam detection is a typical supervised learning problem: the algorithm is fed many emails along with their label (spam or not spam).

**9. What is an online learning system?**

> An online learning system can learn incrementally, as opposed to a batch learning system. This makes it capable of adapting rapidly to both changing data and autonomous  systems, and of training on very large quantities of data.

**10. What is out-of-core learning?**

>Out-of-core algorithms can handle vast quantities of data that cannot fit in a computer's main memory. An out-of-core learning algorithm chops the data into mini-batches and uses online learning techniques to learn from these mini-batches.

**11. What type of learning algorithm relies on a similarity measure to make predictions?**

>An instance-based learning system learns the training data by heart; then, when given a new instance, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.

**12. What is the difference between a model parameter and a learning algorithm's hyperparameter?**

>A model has one or more model parameters that determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).

**13. What do model-based learning algorithms search for?What is the most common strategy they use to succeed?How do they make
predictions?**

>Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually train such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions, we feed the new instance's features into the model's prediction function, using the parameter values found by the learning algorithm.

**14. Can you name four of the main challenges in Machine Learning?**

>Some of the main challenges in Machine Learning are the lack of data, poor data quality, non-representative data, uninformative features, excessively simple models that underfit the training data, and excessively complex models that overfit the data.

**15. If your model performs great on the training data but generalized poorly to new instances,what is happening?Can you name three
possible solutions?**

>If a model performs great on the training data but generalizes poorly to new instances, the model is likely over-fitting the training data (or we got extremely lucky on the training data). Possible solutions to overfitting are getting more data, simplifying the model (select a simpler algorithm, reducing the number of parameters or features used, or regularized the model), or reducing the noise in the training data. 

**16. What is a test set and why would you want to use it?**

> A test set is used to estimate the generalization error that a model will make on new instances, before the model is launched in production.

**17. What is the purpose of a validation set?**

>A validation set is used to compare models. It makes it possible to select the best model and tune the hyperparameters.

**18. What can go wrong if you tune hyperparameter using the test set?**

>If you tune hyperparameters using the test set, you risk overfitting the test set, and the generalization error you measure will be optimistic (you may launch a model that performs worse than you expect).

**19. What is cross-validation and why would you prefer it to a validation set?**

>Cross-validation is a technique that makes it possible to compare models (for model selection and hyperparameter tuning) without the need for a separate validation set. This saves precious training data.
------

 ## :dart:Checklist for Machine Learning project

This checklist can guide you through your Machine Learning projects. There are eight main steps:

- [ ] [Frame the problem and look at the big picture](#Frame-the-Problem-and-Look-at-the-Big-Picture).

- [ ] [Get the data](#Get-the-Data).

- [ ] [Explore the data to gain insights](#Explore-the-Data).

- [ ] [Prepare the data to better expose the underlying data patterns to Machine Learning algorithms](#Prepare-the-Data).

- [ ] [Explore many different models and short-list the best ones](#Short-List-Promising-Models).

- [ ] Fine-tune your models and combine them into a great solution.

- [ ] Present your solution.

- [ ] Launch, monitor, and maintain your system.

### Frame the Problem and Look at the Big Picture

1. Define the objective in business terms.

2. How will your solution be used?

3. What are the current solutions/workarounds (if any)?

4. How should you frame this problem (supervised/unsupervised, online/offline, etc.)?

5. How should you performance be measured?

6. Is the performance measure aligned with the business objective?

7. What would be the minimum performance needed to reach the business objective?

8. What are comparable problems? Can you reuse experience or tools?

9. Is human expertise available?

10. How would you solve the problem manually?

11. List the assumptions you (or others) have made so far.

12. Verify assumptions if possible.

### Get the Data

**Note:** automate as much as possible so you can easily get fresh data.

1. List the data you need and how much you need.
2. Find and document where you can get that data.
3. Check how much space it will take.
4. Check legal obligations, and authorization if necessary.
5. Get access authorizations.
6. Create a workspace (with enough storage space).
7. Get the data.
8. Convert the data to a format you can easily manipulate (without changing the data itself).
9. Ensure sensitive information is deleted or protected (e.g., anonymized).
10. Check the size and type of data (time series, sample, geographical, etc.).
11. Sample a test set, put it aside, and never look at it (no data snooping!).

### Explore the Data

**Note:** try to get insights from a filed expert for these steps.

1. Create a copy of the data for exploration (sampling it down to a manageable size if necessary).

2. Create a Jupyter notebook to keep a record of your data exploration.

3. Study each attribute and its characteristics:

   - Name

   - Type (categorical, int/float, bounded/unbounded, text, structured, etc.)

   - % of missing values

   - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)

   - Possibly useful for the task?

   - Type of distribution (Gaussian, uniform, logarithmic, etc.)

4. For supervised learning tasks, identify the target attribute(s).

5. Visualize the data.

6. Study the correlations between attributes.

7. Study how you would solve the problem manually.

8. Identify the promising transformations you may want to apply.

9. Identify extra data that would be useful.(go back to "[Get the Data](#Get-the-Data)").

10. Document what you have learned.

### Prepare the Data
**Notes:**

- Work on copies of the data (keep the original dataset intact).

- Write functions for all data transformations you apply, for five reasons:
  — So you can easily prepare the data the next time you get a fresh datasets

  — So you can apply these tranformations in future projects

  — To clean and prepare the test set

  — To clean and prepare new data instances once your solution is live

  — To make it easy to treat your preparation choices as hyperparameters

1. **Data cleaning:**

   - Fix or remove outliers (optional).

   - Fill in mission values (e.g., with zero, mean, median...) or drop their rows (or columns).

2. **Feature selection (optional):**

   - Drop the attributes that provide no useful information for the task.

3. **Feature engineering, where appropriate:**

   - Discretize continuous features.
   - Decompose feature (e.g., categorical, date/time, etc.).
   - Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.).
   - Aggregate features into promising new features.

4. **Feature scaling: standardize or normalize features.**

### Short-List Promising Models

**Notes:**

- If the data is huge, you may want to sample smaller training sets so you can train many different models in a reasonable time (be aware that this penalizes complex models such as large neural nets or Random Forests).

- Once again, try to automate these steps as much as possible.

1. Train many quick and dirty models from different categories (e.g., linear, naive Bayes, SVM, Random Forests, neural net, etc.) using standard parameters.

2. Measure and compare their performance.

   - For each model, use N-fold cross-validation and compute the mean and standard deviation of the performance measure on the N folds.

3. Analyze the most significant variables for each algorithm.

4. Analyze the types of errors the models make.

   - What data would a human have used to avoid these errors?

5. Have a quick round of feature selection and engineering.

6. Have one or two more quick iterations of the five previous steps.

7. Short-list the top there to five most promising models, preferring models that make different types of errors.

### Fine-Tune the System
