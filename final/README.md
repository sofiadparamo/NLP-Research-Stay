# Research Stay - Final Project
## Introduction

In natural language processing, accurately detecting and classifying emotions from textual data is a pivotal challenge, with far-reaching applications in sentiment analysis, customer feedback interpretation, and human-computer interaction. This final project is designed to provide students with hands-on experience in this intricate domain of machine learning. The task involves the preparation, analysis, and emotion annotation of a text dataset, employing three distinct computational approaches: rule-based, neural networks, and deep learning.

## Objectives
The students will apply three different approaches to emotion detection: 

 - Rules-based processing. 
 - Neural Networks. 
 - Deep Learning.

The target deliverable is a written final report that includes the following characteristics:

 - Title
 - Abstract
 - Motivation
 - Literature review
 - Experimentation
 - Results
 - Discussion
 - Conclusions
 - Bibliography

## Description

The students must download and execute the proportionated Python program capable of processing the prepared dataset. The program is adept at training models based on the specified approaches and generating emotion detection predictions. 

 - The rule-based method will involve predefined rules and lexicons to
   infer emotions from the text.
 - The neural network approach will utilize a more traditional machine
   learning algorithm, leveraging the power of artificial neural
   networks.
 - The deep learning approach will involve implementing more advanced
   and layered neural networks, allowing for a more intricate
   understanding of textual data.

The core objective of this project is to enable students to discern the nuances, strengths, and limitations inherent in each method. A comprehensive analysis of the performance differences among these algorithms will provide valuable insights into their respective efficiencies and applicability in various scenarios.

The deliverable for this project is a formal report with a critical analysis of the performance of each algorithm. This report should serve as a testament to the students' understanding and ability to critically evaluate machine learning techniques in the context of text emotion detection.

## Abstract

An abstract is a concise summary of a research paper or thesis. It serves as a snapshot of the main aspects of the research work, providing readers with a quick overview of the study. It typically includes the research problem, objectives, methodology, key findings, and conclusions. An effective abstract lets readers quickly ascertain the paper's purpose and decide whether the rest of the document is worth reading.

For this project, the abstract must comply with the following requirements:

-   **Start with a Clear Purpose**: Begin by clearly stating the main aim or problem addressed by the research. This helps set the stage for the readers.
-   **Describe the Methodology**: Briefly explain the methods used to carry out the research. This gives readers a glimpse into how the study was conducted.
-   **Summarize Key Findings**: Highlight the main findings or results of the research. This should clearly present the significant outcomes with a manageable amount of detail.
-   **Conclude with the Impact**: End the abstract with the implications or significance of the findings. This is where you indicate the contribution of your research to the field.

The abstract must:

-   Be concise, typically within 150-250 words.
-   Stand alone, meaning it should be understandable without reading the full paper.
-   Avoid using jargon or acronyms that are not widely known.
-   Not contain citations or references.
-   Provide a complete overview, including the purpose, methods, results, and conclusions.

Example:

**Title**: Leveraging Machine Learning for Enhanced Emotion Detection in Textual Data

**Abstract**: This study investigates the application of machine learning techniques in detecting emotions from textual data. Given the growing interest in understanding affective states in online communication, this research aims to advance the field by developing a more accurate emotion detection model. Using a dataset of over 10,000 annotated texts, we employed traditional machine learning algorithms and deep learning approaches, specifically convolutional neural networks (CNNs), to classify texts into six primary emotions: joy, sadness, anger, fear, surprise, and love. The methodology included pre-processing textual data, feature extraction, model training, and validation. Our findings reveal that while traditional algorithms like Support Vector Machines (SVM) provided a solid baseline, CNNs demonstrated superior performance in terms of accuracy, achieving a 12% improvement over the SVM model. The study concludes that deep learning, with its ability to capture complex patterns in data, holds significant promise for enhancing emotion detection in textual content. These findings have implications for various applications, from improving mental health interventions to refining customer service interactions in the digital space.

## Motivation

The motivation section of a research paper is where you justify the necessity of your study. It's the "why" behind your research. This section explains the importance of the problem you are addressing, the gap in existing research that your study intends to fill, and the potential impact of your findings. Essentially, it answers the question: Why does this research matter?

For this project, the motivation must comply with the following requirements:

-   **Identify the Problem**: Begin by clearly defining the problem or issue your research addresses. This sets the stage for explaining why your study is essential.
    
-   **Review Existing Literature**: Briefly discuss what has already been done in this area and identify the gaps or limitations in the current knowledge. This shows that there is a need for your research.
    
-   **Explain the Significance**: Clarify why filling the identified gap is important. This could be due to its theoretical, practical, or societal implications.
    
-   **State the Objectives**: Clearly outline what your research aims to achieve. This links back to the identified problem and the significance of solving it.
    
-   **Be Concise and Focused**: The motivation section should be to the point, avoiding unnecessary details.

Example:

The proliferation of digital communication has led to an exponential increase in textual data, making it imperative to understand the emotional undercurrents in these interactions. While emotion detection has been explored, the accuracy and depth of these analyses still need to be improved, especially in diverse and nuanced contexts. Existing models often struggle with subtleties in language, cultural differences, and varied expressions of emotions. This gap has significant implications, as accurate emotion detection is crucial for applications ranging from mental health monitoring to customer service optimization.

Our research is motivated by the need to enhance the understanding of emotions in textual data, leveraging the advancements in machine learning. We aim to address the shortcomings of current models by implementing both traditional machine learning algorithms and cutting-edge deep learning techniques. The goal is to create a model that improves accuracy and adapts to the complexities and subtleties of human emotions in textual communication. This research can potentially revolutionize how we interact with and interpret textual data, offering profound implications for various sectors, including healthcare, marketing, and social media.

## Literature Review

The literature review section of a research paper provides an overview of existing research related to your study. It involves critically analyzing and synthesizing previous studies to establish a foundation for your research. This section demonstrates your understanding of the field, highlights progress, and identifies where your research fits into the existing body of knowledge.

For this project, the Literature Review must comply with the following requirements:

-   **Define the Scope**: Clearly outline the boundaries of your review. Focus on literature that is directly relevant to the research.
    
-   **Organize the Review**: Structure your review logically. You can organize it chronologically, thematically, or methodologically, depending on what makes the most sense for your topic.
    
-   **Summarize and Synthesize**: For each work, provide a brief summary and discuss how it contributes to the field. Then, synthesize the findings to show trends, conflicts, or gaps in the research.
    
-   **Critically Evaluate**: Offer a critical analysis of the literature. Discuss the strengths and weaknesses of previous studies and methodologies.
    
-   **Link to Your Study**: Explain how the literature review leads to your research question or hypothesis. Highlight the gap this study aims to fill.

-   **Required bibliography**: At least 7 research papers must be cited and compared for this project. These papers can be selected from the pool of documents provided in previous weeks or can be other papers found in peer-reviewed journals.

Example:

Emotion detection in textual data has been an evolving area of research within machine learning and natural language processing. Early attempts primarily employed rule-based methods and lexical approaches, as seen in studies by Smith et al. (2010) and Jones et al. (2012), which focused on identifying keywords indicative of emotional states. While these approaches provided a foundation, they needed more contextual understanding and flexibility.

The advent of machine learning algorithms brought significant advancements. Research by Lee and Kim (2015) demonstrated the potential of Support Vector Machines (SVM) in classifying emotions, achieving notable accuracy. However, these models often struggled with the complexity and subtleties of natural language.

Recent studies have shifted towards deep learning techniques, particularly Convolutional Neural Networks (CNNs). A groundbreaking study by Zhang et al. (2018) showcased the ability of CNNs to capture intricate patterns in textual data, significantly enhancing emotion detection accuracy. Despite these advancements, challenges remain in dealing with cultural nuances, linguistic variations, and implicit emotional expressions.

Our research builds upon these findings, aiming to address the limitations of current models by integrating both traditional and deep learning approaches. We seek to develop a more robust and adaptable emotion detection model by exploring the synergies between different methodologies.


## Experimentation:

The experimentation section of a research paper describes the practical steps you took to investigate your research question. It should detail the methodologies used, the experimental setup, the data, and the procedures followed. This section must provide enough detail for another researcher to replicate your study.

For this project, the Experimentation section must comply with the following requirements:

-   **Methodology Description**: Clearly explain the methodologies used in your study. In this case, describe the three different Python programs for emotion detection, each representing a different approach (rule-based, neural networks, deep learning).
    
-   **Experimental Setup**: Outline the setup for the experiments, including the datasets used (train, test, and validation sets), and the hardware and software configurations.
    
-   **Procedure**: Detail the procedure followed in the experiments. Mention that each program has various hyperparameters which can be adjusted, and that students are expected to execute each algorithm at least 10 times with different hyperparameter values.
    
-   **Data Collection and Analysis**: Explain how the data (in this case, accuracy metrics) will be collected and analyzed. Mention that students are expected to graph the changes in accuracy to visualize the results.
    
-   **Flow Diagrams**: State that students are expected to create flow diagrams to explain how each program works and how the algorithms learn.
    
-   **Replicability**: Ensure that the description is thorough enough to allow replication of the study by others.

Activities to be performed for this section:

**Experimentation:** Three distinct Python programs have been provided, each embodying a different approach to emotion detection: rule-based, neural networks, and deep learning. The programs are designed to analyze a labeled text dataset to predict emotions.

**Methodology:** Each program is equipped with adjustable hyperparameters, allowing for the fine-tuning of the algorithms to optimize accuracy. Students are provided with the source code and are expected to **run each program at least 10 times**, altering hyperparameters in each iteration.

**Experimental Setup:** The dataset is divided into training, testing, and validation sets. The provided python programs already use the training set for initial model training, the test set for evaluating performance, and the validation set for final accuracy assessment.

**Procedure:** Students will systematically adjust hyperparameters such as learning rate, batch size, or layers in the neural network, and **document the resultant changes in accuracy**. These variations aim to explore the impact of each parameter on the model’s performance.

**Data Collection and Analysis:** The accuracy of each run is recorded, **and students are required to graph these results to visualize the impact of hyperparameter adjustments.** This graphical representation will aid in understanding the correlation between hyperparameters and model performance.

**Flow Diagrams:** Additionally, students are expected to **create flow diagrams that delineate the operational process of each program, elucidating how the input data is processed and how the algorithm learns and predicts emotions**. These diagrams must be high-level, the objective here is not to produce an algorithm, but to show understanding of the moving parts in each approach.


## Results

The results section of a research paper is where you present the findings of your study without interpretation. It's a factual report of the data collected during your experiments. This section should be clear, concise, and objective, allowing readers to understand the outcomes of your research.

For this project, the Results section must comply with the following requirements:

-   **Present Findings Clearly**: Report the findings from your experiments logically and systematically. You can use figures, tables, and graphs to show your data where appropriate.
    
-   **No Interpretation**: Avoid discussing the implications or significance of your results in this section. Focus solely on presenting the data.
    
-   **Accuracy and Detail**: Provide precise measurements and avoid omitting relevant data. Details such as sample sizes, response rates, and statistical significance should be included.
    
-   **Organize Sequentially**: Present your results in the same sequence that you described the methodology in the experimentation section.

Example:

**Results:**

**Rule-Based Approach:** In the rule-based emotion detection approach, the average accuracy across 10 trials with varying hyperparameters was 58%. The highest recorded accuracy was 62%, achieved with a specific set of lexical rules. Figure 1 illustrates the changes in accuracy as hyperparameters were adjusted.

**Neural Networks:** The neural network approach yielded more promising results, with an average accuracy of 75%. The peak accuracy achieved was 81%, attributed to optimal hyperparameter tuning. Figure 2 demonstrates the fluctuation in accuracy with different hyperparameter configurations.

**Deep Learning:** The deep learning model demonstrated the highest efficacy, with an average accuracy of 85%. The maximum accuracy recorded was 90%, observed with an intricate layer configuration and learning rate adjustment. Figure 3 depicts the accuracy trends throughout the trials.

**Comparative Analysis:** A comparative analysis of all three approaches is presented in Figure 4, highlighting the varying levels of accuracy achieved in each method. This comparison underscores the differences in performance among the algorithms.

**Flow Diagrams:** Accompanying the quantitative results, the following diagrams provided a visual representation of how the algorithms processed input data and predicted emotions.


## Discussion

In the discussion section, you interpret and analyze your findings, contextualizing them within the broader field of study. This section should explore the implications of your results, discuss any unexpected findings, and relate your research to existing knowledge.

For this project, the discussion section must comply with the following requirements:

-   **Interpret Results**: Offer interpretations of what your findings mean. Explain how your results relate to your hypotheses or research questions.
    
-   **Contextualize with Literature**: Relate your findings to existing studies and literature. Discuss how your results support, challenge, or extend current knowledge.
    
-   **Discuss Limitations**: Acknowledge any limitations or weaknesses in your study and suggest how they might be addressed in future research.
    
-   **Implications and Recommendations**: Highlight the implications of your findings and suggest practical applications or recommendations for future research.

For this section, it is expected that the student will give an educated opinion about the changes in accuracy for each algorithm and also the impact that the change in hyperparameters makes on performance. In addition to the previous points **the student must respond to the following questions:** 

## *If the hyperparameters are the key to 100% accuracy, why not just increase these hyperparameters until perfection in accuracy is achieved? What is preventing computer science from doing that?*

Example:

Our study's findings indicate significant variations in the performance of different emotion detection algorithms. The rule-based approach, while simpler, offered lower accuracy (average 58%) compared to neural networks and deep learning models. This aligns with the inherent limitation of rule-based systems, which struggle to capture the nuances and complexities of human emotions.

The neural network approach demonstrated better performance (average 75%), suggesting that machine learning models can effectively learn from data to predict emotions. However, the highest accuracy was achieved by the deep learning model (average 85%), emphasizing the potential of deep learning in processing and interpreting textual data.

Interestingly, the deep learning model's peak accuracy (90%) highlights the importance of hyperparameter tuning and model architecture in optimizing performance. This finding is consistent with literature emphasizing the critical role of hyperparameters in machine learning models (Smith, 2021).

A noteworthy limitation of our study is the reliance on a single dataset. Future research should consider diverse datasets to validate the generalizability of our findings.

The implications of our study are substantial for fields like sentiment analysis and human-computer interaction, where accurate emotion detection is paramount. We recommend further exploration into deep learning architectures and hyperparameter optimization to enhance emotion detection algorithms.

Our research contributes to the growing body of knowledge on emotion detection in textual data. It underscores the importance of choosing appropriate machine learning models and tuning them for optimal performance.

## Conclusions

The conclusions section is the final part of your research paper where you summarize your findings, restate the significance of your research, and provide a closure. This section should encapsulate the essence of your study, offering a concise synthesis of the key points.

For this project, the conclusions section must comply with the following requirements:

-   **Summarize Key Findings**: Briefly reiterate the main findings of your research. This is not a place for introducing new information.
    
-   **Restate Research Significance**: Highlight the importance of your study and its contribution to the field.
    
-   **Reflect on Limitations**: Quickly recap the limitations of your study, acknowledging its scope and boundary conditions.
    
-   **Suggest Future Research**: Offer recommendations for future studies that can build upon your work.

Example:

Our study embarked on a comparative analysis of three distinct emotion detection algorithms: rule-based, neural networks, and deep learning. The findings revealed that while rule-based approaches provide a foundational level of accuracy (average 58%), they are surpassed by machine learning models. Neural networks exhibited improved accuracy (average 75%), showcasing their learning capabilities. However, deep learning models emerged as the most effective, attaining the highest average accuracy of 85%.

These results underscore the progressive enhancement in emotion detection accuracy as we move from rule-based methods to more sophisticated machine learning and deep learning techniques. The study highlights the critical role of algorithm selection and hyperparameter tuning in achieving optimal performance in emotion detection tasks.

While our research provided valuable insights, it was limited by the use of a single dataset. Future studies should consider a more diverse range of datasets to validate and extend our findings.

In conclusion, our research contributes to the evolving landscape of emotion detection in textual data. It emphasizes the potential of machine learning, particularly deep learning, in deciphering complex human emotions. This study serves as a stepping stone for future research aimed at refining emotion detection algorithms and exploring their applications in various domains.

## Bibliography

For this project, the bibliography must be included and formatted according to the LaTeX template in the repository.

## Format

A specific LaTeX format has been provided for this project (ITESM-MIT-MasterThesis.zip). This format is designed to ensure consistency and professionalism in the presentation of your reports. LaTeX is a high-quality typesetting system widely used for scientific and technical documents due to its precise control over layout and formatting.

Download the provided template and import it into your Overleaf.com account. Write your report and your bibliography according to this template. 

Your final submission will be done through your personal github account and an email notice.
