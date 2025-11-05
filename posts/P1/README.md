# The rules of Machine Learning and Rule-Based Systems

## Situations where Rule-Based Systems might rule

![](ifthen.png)

Every so often, I see a post on social media about someone who created enormous value with data solutions built solely on simple business rules. I am somewhat skeptical of this claim. After all, if it is that simple, why don’t we have flying cars by now?

Businesses are complex sociotechnical systems embedded in even more complicated and competitive socio-economic-political systems, with access to only partial information.

Therefore, the skeptical side of me has a hard time believing a single person can come up with a revolutionary, straightforward, never-before-thought-of solution in a short period of time. However, I think there’s a lesson to learn here, and we should remember Google’s first rule of Machine Learning:

```         
Rule #1: Don’t be afraid to launch a product without machine learning.

Machine learning is cool, but it requires data. Theoretically, you can take data from a different problem and then tweak the model for a new product, but this will likely underperform basic heuristics. If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there.

For instance, if you are ranking apps in an app marketplace, you could use the install rate or number of installs as heuristics. If you are detecting spam, filter out publishers that have sent spam before. Don’t be afraid to use human editing either. If you need to rank contacts, rank the most recently used highest (or even rank alphabetically). If machine learning is not absolutely required for your product, don’t use it until you have data.
```

The Rules are somewhat outdated, since we now have better techniques for Transfer Learning and In-Context Learning. We also always had not mentioned, less data-hungry tools, like Mathematical Optimization or Simulation. However, there’s still a lot of sound advice in the guide.

The easier-to-see examples where heuristics can be incredibly effective usually fall within the realm of Recommender Systems. If the user has finished watching a particular episode of a TV show on Netflix, the show’s next episode is a natural recommendation.

In fact, when I was still in college, we had a Kaggle-style RecSys competition with an unusual application: recommend countries that were more appropriate for a soccer player to represent.

Naturally, if you know anything about soccer, you know that most players represent the country of their birth. While many of my classmates tried to come up with complex algorithms, this rule proved quite challenging to beat, and it was part of the solution that got me the win.

Vincent Warmerdam has delightful techniques to speed up the creation of your Rule-Based System that require less expert knowledge, and I have used them in the past with some success.

For clarification, I am using a loose definition of a Rule-Based System here; I am not talking about formal Expert Systems or Association Rule Learning, even though I could. As an example, think of it as a SQL query that your team created together with a business expert.

As with everything else, Rule-Based Systems have limitations:

```         
They require expert knowledge to be effective.

They are hard to scale: your rules can get too complex to maintain if you try to cover every possible corner case.

When Concept Drift occurs, it’s not as easy to fix as simply starting a retraining pipeline.

Sometimes things ain’t easy, and not all Machine Learning is predictive: tasks like Causal Inference usually require sophisticated estimation procedures, and simple heuristics ain’t gonna cut it.
```

Therefore, let’s not forget about Google’s rule #3 of Machine Learning:

```         
Rule #3: Choose machine learning over a complex heuristic.

A simple heuristic can get your product out the door. A complex heuristic is unmaintainable. Once you have data and a basic idea of what you are trying to accomplish, move on to machine learning. As in most software engineering tasks, you will want to be constantly updating your approach, whether it is a heuristic or a machine­-learned model, and you will find that the machine­-learned model is easier to update and maintain (see Rule #16).
```

That being said, Rule-Based Systems also have fantastic upsides:

```         
They serve as a baseline for your ML System, often a strong one.

They serve as a Fallback System in case of issues with your ML model.

During creation, they force you to think carefully about the problem, serving as a form of Exploratory Data Analysis.

Its predictions are usually super simple to explain.

Later, you can combine or incorporate them as features into your Machine Learning Model. See Rule #7: Turn heuristics into features, or handle them externally.

As a matter of fact, rules can serve as the basis for an entire ML approach: Weak Supervision.
```

Take-home message: Rule-based systems provide strong baselines for predictive Machine Learning tasks and should come up more frequently, especially if you’re starting a project from scratch with little prior knowledge about your company or industry standards. They force you to think carefully not only about the data but also about the business as a whole. Later, you can incorporate them into your Machine Learning System via Feature Engineering or as a Fallback System for Disaster Recovery. However, they require expert knowledge to create and can become hard to maintain, scale, and adapt. Eventually, as your analytical maturity evolves, you will be capable of moving beyond them.

References

```         
Google Rules of Machine LearningGoogle Rules of Machine Learning

Vincent Warmerdam - Keynote “Natural Intelligence is All You Need [tm]”:

Vincent Warmerdam - Playing by the Rules-Based-Systems | PyData Eindhoven 2020:
```