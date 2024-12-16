# **Word Translation Without Parallel Data**

This repository contains the implementation and related materials for the paper *"Word Translation Without Parallel Data"* by A. Conneau, G. Lample, M. Ranzato, L. Denoyer, and H. Jégou, published in 2018. The paper proposes an **unsupervised method for word translation** that achieves state-of-the-art performance without requiring parallel corpora. The authors use adversarial training to align monolingual word embeddings across languages and introduce **Cross-Domain Similarity Local Scaling (CSLS)** to improve translation accuracy.

### **Repository Structure**
- **`main` branch:** Contains the core setup for the project.
- **`implementation` branch:** Includes the code for re-implementing the proposed method.
- **`poster-presentation` branch:** Hosts the materials related to the poster presentation.
- **`final` branch:** Contains the final submitted code.

### **Poster Presentation**
You can view the poster summarizing the research paper by clicking the link below:

[View Poster (PDF)](https://github.com/JakubCiesko/ML-NLP-WTWPD/blob/poster-presentation/poster_no_intro.pdf)

For convenience, here's a preview of the poster:

![Poster Preview](./poster_preview.png)

---

### **Paper Details**
- **Title:** Word Translation Without Parallel Data  
- **Authors:** A. Conneau, G. Lample, M. Ranzato, L. Denoyer, H. Jégou  
- **Publication Year:** 2018  
- **Published on:** arXiv  
- **DOI:** [10.48550/arXiv.1710.04087](https://doi.org/10.48550/arXiv.1710.04087)  
- **Code Implementation:** [facebookresearch/MUSE](https://github.com/facebookresearch/MUSE)  

### **Key Ideas**
1. **Adversarial Training:** Align monolingual word embeddings from different languages without requiring parallel data by using a discriminator to classify embeddings.
2. **Mapping Objective:** Use a linear mapping to minimize differences between aligned embeddings across languages.
3. **Cross-Domain Similarity Local Scaling (CSLS):** Reduces the hubness problem in high-dimensional spaces to improve translation quality.
4. **Evaluation:** Achieves competitive results across multiple language pairs, even in low-resource settings.

---
