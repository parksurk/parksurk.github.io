---
title: "Digital Treasure Quest: Our Journey in Dialogue Summarization"
excerpt: "Our team, Digital Treasure Quest, participated in a competition focused on summarizing daily conversations, experimenting with various deep learning models like KoBART and T5. Each member contributed by improving models through hyperparameter tuning, data augmentation, and using translation to enhance summarization performance. The model's effectiveness was evaluated using ROUGE metrics, and the ensemble model combining KoBART and T5 achieved the best results. Despite facing several challenges, we overcame them through collaboration and the use of AI tools, leading to the successful submission of our final model. This experience deepened our understanding of NLP and machine learning applications and highlighted the importance of teamwork in research and development."
date: 2024-09-23 12:00:10 -0400
toc: true
toc_sticky: true
categories:
  - AIBootcamp
tags:
  - AIBootcamp 
  - 패스트캠퍼스 
  - 패스트캠퍼스AI부트캠프 
  - 업스테이지패스트캠퍼스 
  - UpstageAILab 
  - 국비지원 
  - 패스트캠퍼스업스테이지에이아이랩 
  - 패스트캠퍼스업스테이지부트캠프
---


# **Our Journey in Dialogue Summarization: Team Digital Treasure Quest**

As part of **Team Digital Treasure Quest**, I had the opportunity to collaborate with talented peers: **Baek Kyung-tak, Han Ah-reum, and Wi Hyo-yeon**. Our team aimed to tackle one of the most challenging tasks in Natural Language Processing (NLP): **Dialogue Summarization**. The project was part of a competition where we sought to develop models that can extract key points from everyday conversations.

## GitHub
https://github.com/parksurk/nlp-plm-baseline

## **1. Competition Overview**
The goal of the competition was to create models capable of summarizing daily dialogues efficiently. Dialogue Summarization can greatly enhance productivity, especially in environments where meetings and everyday exchanges occur frequently, but revisiting entire conversations is time-consuming.

### **Dataset**
- **Training set**: 12,457 dialogues and their summaries.
- **Validation set**: 499 dialogues and summaries.
- **Test set**: 250 dialogues (hidden for final evaluation).

The task was to train models that could predict the summaries based on dialogues involving 2-7 participants. We experimented with **deep learning models** like **BART, T5, and GPT**, aiming for high **ROUGE** scores, a standard metric for evaluating the quality of generated summaries by comparing them to reference texts.

## **2. Team Structure and Collaboration**
Our team's composition provided unique advantages:
- **Strengths**: Diverse viewpoints, openness to AI assistant tools, and the ability to experiment with different methodologies.
- **Challenges**: Limited experience in collaborative R&D using Git, Python, and machine learning domains.

## **3. Strategy and Workflow**
### **3.1. Common Baseline Development**
We started by developing a **baseline model** that the whole team could build upon. This allowed us to ensure consistency in our experiments while giving each team member the flexibility to explore different avenues for model improvement.

### **3.2. Modeling Improvements**
- **Model Selection**: We explored **Transformer-based models** like **BART** and **T5**, as well as advanced tokenization strategies to handle the nuances of Korean dialogue.
- **Algorithm Optimization**: Hyperparameter tuning and experimenting with various **learning rates**, **batch sizes**, and **decoder lengths** helped improve the model performance.

### **3.3. Key Experiments**
Each team member conducted unique experiments:
- **Baek Kyung-tak** tested different model architectures, discovering that **KoBART** performed best for Korean text summarization.
- **Han Ah-reum** explored **data augmentation** techniques, focusing on tokenization methods and analyzing dialogue lengths.
- **Wi Hyo-yeon** applied ensemble methods, combining models like **T5** and **KoBART**, leading to significant performance boosts.
  
## **4. Challenges and Learnings**
We faced several obstacles, including:
- **Computational Constraints**: Long training times (up to 20 hours per run) slowed down experimentation. Memory limitations also restricted us from using more advanced models.
- **Submission Errors**: Some of our top-performing models encountered submission errors due to format issues.

Despite these challenges, we leveraged tools like **ChatGPT** and **DeepL** to help refine translation and summarization processes, enhancing our efficiency.

## **5. Final Results**
Ultimately, our approach of blending **model optimization** with **collaborative experimentation** yielded positive results. The ensemble model combining **T5** and **KoBART** produced the best scores, with improvements in **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** scores.

## **6. Key Takeaways**
- **Collaboration**: The diverse skill sets of our team members proved invaluable. Each person’s contributions, whether in model tuning, data preprocessing, or code optimization, played a crucial role.
- **Iteration**: Frequent experimentation and a willingness to embrace both successes and failures were key to improving our model’s performance.
- **Adaptability**: Using AI tools like **ChatGPT** and **DeepL** allowed us to overcome limitations and develop efficient processes for dialogue summarization.


This experience has been a significant learning journey, as I sharpened my machine learning skills and gained deeper insights into NLP techniques. The **Dialogue Summarization** competition not only pushed us to develop cutting-edge dialogue summarization models but also highlighted the importance of collaboration, iteration, and adaptability in real-world AI challenges.
