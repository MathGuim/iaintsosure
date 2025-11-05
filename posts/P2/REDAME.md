# Toyota Production System: Rudimentary Causality in Industrial Engineering

## A simplified exercise on History of Ideas

Even though the Toyota Production System for manufacturing started being devised in the 1940s, it still offers many valuable lessons for Data Scientists, ML, AI and Data Engineers, and the Technology sector as a whole.

Someday, I will post on how to approach concepts such as Hansei, Kaizen, Genchi Genbutsu, and Jidoka in our own field.

Brazil came into contact with many of these ideas through the works of Prof. Vicente Falconi. In the 1980s, the professor traveled to Japan to study with scientists and engineers at JUSE (the Union of Japanese Scientists and Engineers).

On a small autobiographical note, I don’t come from a traditional data science background, such as Statistics or Computer Science. Even though I have worked on problems that interest Industrial Engineers, such as Demand Planning, and even though the degree gave me invaluable skills that I still make heavy use of today, such as Data Structures and Algorithms, Operations Research, and Statistics. And I nevertheless also had a memorable work experience at Falconi, the company founded by Prof. Falconi, I never had the opportunity to work as an Industrial Engineer.

There are many reasons why the majority of Brazilian engineering graduates end up not working in their fields. I might discuss some in a future post, even though my case is a bit unusual.

Regardless, I used to have another informal occupation, working as a Historian of Ideas, and few things excite me more than seeing how ideas evolve. I used to be fascinated by the history of Mathematics, for example.

But returning to today’s topic, many of the Japanese engineers had a solid understanding of Statistics. They made heavy use of Design of Experiments and Statistical Process Control, for example.

But the original quest, the one they were obsessed with was the quest to know the reason “why?”. They wanted to know the root causes of problems that hindered organizational learning and, therefore, impeded continuous improvement (Kaizen).

To tackle this herculean task, they adapted and developed many ideas that ultimately led to the creation of entire Philosophies and sets of techniques, from which I am going to talk about three today: 5W, 5M, and Ishikawa Diagrams.

The first one is called the 5W: the basis of Toyota’s scientific approach is to ask why five times whenever you find a problem. In the Toyota Production System, “5W” refers to the 5 Whys. By repeating why five times, the nature of the problem as well as its solution becomes clear, the engineers thought. The solution, or the how-to, is designated as “1 H”. Thus, “Five Why’s equal One How” (5W = 1H).

In the 1960s, Ishikawa introduced and popularized a new technique for identifying root causes, the Fishbone Diagram. Ishkawa had supposedly been inspired by a talk given by W. Edwards Deming. However, according to Wikipedia, the idea had been around since the 1920s; unfortunately, the article doesn’t cite a source, and I don’t have time to do a complete investigation.

To fill in the spine of the Fishbone Diagram, they categorized the five broad classes of causes of issues in the Manufacturing Industry, originating the 5Ms:

```         
Manpower (physical or knowledge work)

Machine (equipment, technology)

Material (includes raw material, consumables, and information)

Method (process)

Measurement (inspection, environment)
```

![alt text](Ishikawa_Fishbone_Diagram.png)

Take a look at one template of an Ishikawa Cause-and-Effect diagram. If it sounds familiar, it’s because it is! It’s a significantly simplified and somewhat distorted form of a Causal Directed Acyclic Graph (DAG). In the next post, I will share my opinion on how Causal DAGs bridge the gap between formalism and intuition. I am going to connect the dots between this post, causality, and an overarching theme throughout this blog — and, in a sense, my life: intellectual humility.

Of course, nowadays we know Root-Cause Analysis is hard work, and simple heuristics like the Fishbone Diagram and 5W can only help so much. If you want to learn more about Root-Cause Analysis, the DoWhy library has interesting examples for you to get started. For a fun introduction to some of the history and ideas of Causal Inference, you can check Judea Pearl’s The Book of Why. See References.

Take-Home idea: Industrial Engineers, such as Kaoru Ishikawa and Taiichi Ohno, began developing a rudimentary Conceptual Framework of Causality starting from the 1940s, even using proto-causal diagrams later. That’s approximately 20 years after the geneticist Sewall Wright’s Path Diagrams and 60 years before Judea Pearl’s seminal work: Causality! Whatever your opinion of Japan or Toyota today, you can’t deny their success throughout the 20th century!

References

```         
Toyota Production System: Beyond Large-Scale Production

Causality: Models, Reasoning and Inference

The Book of Why: The New Science of Cause and Effect

Finding Root Causes of Changes in a Supply Chain

Finding the Root Cause of Elevated Latencies in a Microservice Architecture
```