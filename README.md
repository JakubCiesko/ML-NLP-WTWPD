# Word Translation Without Parallel Data

This branch contains the code implementation of the paper **"Word Translation Without Parallel Data"**. It includes methods for aligning monolingual word embeddings using adversarial training and the Cross-domain Similarity Local Scaling (CSLS) metric.

Additional files for downloading word vectors and other necessary data will be added.

---

## TODO List

### Core Implementation
- [x] Integrate adversarial training for mapping embedding spaces.
- [ ] Implement the CSLS metric to improve nearest-neighbor search.
- [ ] Add iterative Procrustes refinement for alignment.

### Data and Preprocessing
- [x] Provide scripts to download pre-trained word embeddings.
- [x] Include tools to convert embeddings to the required format.
- [x] Add code to preprocess and normalize embedding spaces.

### Experiments and Extensions
- [ ] Test the impact of different initialization methods for adversarial training.
- [ ] Compare the performance of CSLS with alternative similarity metrics.
- [ ] Evaluate alignment performance across diverse language pairs.

### Additional Features
- [ ] Include configurations for low-resource language scenarios.
- [ ] Add support for multilingual alignment pipelines.
- [ ] Extend to subword or character-level embeddings for morphologically rich languages.

---

