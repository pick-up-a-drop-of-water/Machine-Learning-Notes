# some of most important concepts in Machine Learning

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

**13. What do model-based learning algorithms search for?What is the most common strategy they use to succeed?How do they make  predictions?**

>Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually train such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions, we feed the new instance's features into the model's prediction function, using the parameter values found by the learning algorithm.

**14. Can you name four of the main challenges in Machine Learning?**

>Some of the main challenges in Machine Learning are the lack of data, poor data quality, non-representative data, uninformative features, excessively simple models that underfit the training data, and excessively complex models that overfit the data.

**15. If your model performs great on the training data but generalized poorly to new instances,what is happening?Can you name three  possible solutions?**

>If a model performs great on the training data but generalizes poorly to new instances, the model is likely over-fitting the training data (or we got extremely lucky on the training data). Possible solutions to overfitting are getting more data, simplifying the model (select a simpler algorithm, reducing the number of parameters or features used, or regularized the model), or reducing the noise in the training data. 
