# Complete so far

- 2nd revision of lit review (more to add as research is ongoing)
- Research on data augmentation.

# Feb

## Practical

1. [x] Complete raw drum data collection.
2. [x] Complete data augmentation techniques and preparation for the NN.
3. [x] Convert as much raw data as is needed into augmented data.
4. [x] Seperate this data into training and testing data.
5. [x] Design a NN with justification for each layers usage from research.
6. [ ] Test as proof of concept with some data

## Research

7. [x] In depth research on NN layers and structures:

- [x] Pooling layers (max/min/avg (local & global)).
- [x] Dropout layers
- [ ] Multiclass output layer (a better name exists, needs research)

## Writing

8. [ ] Update lit review
9. [ ] Start methodology
10. [ ] Start abstract

## Day progress

| Sat 23  | Mon 25 | Tues 26 | Weds 27     | Thurs 28   |
| ------- | ------ | ------- | ----------- | ---------- |
| 1,2,3,4 | 7,5,6  | 8,9,10  | Sup Meeting | March plan |

# March

## Practical

0. [x] Separate data into training and test.
1. [x] Build and test first model.
1. [x] Record training time, loss and accuracy of the first model.
1. [x] Build and test second model.
1. [x] Record training time, loss and accuracy of the second model.
1. [ ] Build and test third model.
1. [ ] Record training time, loss and accuracy of the third model.
       _after 14_
1. [ ] Think about improvements to best model and make that.
1. [ ] Build and test final model.

## Research

7. [x] Tensorflow / TensorBoard / Keras documentation

## Writing

8. [x] Update lit review
9. [x] Start methodology
10. [x] Start abstract
11. [ ] Finish (textually) methodology
12. [ ] Finish (textually) abstract
13. [ ] Create any vector images needed and add to dissertation
14. [ ] Was artificial data of a similar accuracy and loss to real data? (kindof conclusion)
15. [ ] April plan

## Day progress

| Thurs 07 | Tues 12 | Thurs 14 | Friday 15 | Mon 18 | Tues 19 |
| -------- | ------- | -------- | --------- | ------ | ------- |
| 0, 1, 7  | 2, 3    | 8, 9     | 10        | 4, 5   | 6, 11   |

| Weds 20 | Thurs 21 | Fri 22 | Mon 25 | Tues 26 | Weds 27 | Thurs 28 | Fri 29       |
| ------- | -------- | ------ | ------ | ------- | ------- | -------- | ------------ |
| 12,     | 13       | 14     | 15     | 16      | BACKLOG | BACKLOG  | BACKLOG & 17 |

**Key dates:**

- Friday 29 March (personal) deadline for practical part

# April

## Practical

- [x] Build and test third model. **1**
- [x] Record training time, loss and accuracy of the third model. **2**
- [x] Think about improvements to best model and make that. **3**
- [ ] Build and test final model. **4**
- [ ] Demo this running over a few drum beats (broken up with onset detection), comparing predicitons to real. **5**

## Writing

- [x] Lit review **0**
  - Cut lit review and add more relevant/recent stuff
  - Summarize lit review process of why each thing is neccessary.
- [ ] Finish abstract **1**
- [x] Create any vector images needed and add to dissertation **2**
- [ ] Methodology: (finish & proof-read**3**)
  - [x] Raw data collection. **4**
  - [x] Processing and augmentation. **5**
  - [ ] Model designs: **6**
    - Model A (1D amplitude input data, high depth CNN, Inception, Dilation Causal Conv, Tanh activations, Multi-label) **6a**
    - Model B (1D amplitude input data, medium depth CNN, Skips, Dilation Causal conv, Leaky ReLU activations, Onehot labels) **6b**
    - Model C (2D spectrogram input data, ...) **6c**
    - Include diagram of each models archetecture. **6d**
  - Training minimises a loss function by adjusting NN weights to correct output incrementally.
  - Optimisers experimented with: **7**
    - SGD
    - Adam
  - Accuracy = mean(0-1LossF) **8**
  - [x] Tensorflow, collect accuracy and loss per epoch into log files. **9**
    - Binary crossentropy
    - Hamming loss
    - KL divergence
  - [x] Explain confusion matrices. **10**
- [ ] Analysis:
  - [ ] Compare loss and accuracy of models over time (training / validation sets). **11**
  - [x] Test data (augmented) accuracy and loss values. **12**
  - [x] Confusion matrices. **13**
  - [x] Was artificial data of a similar accuracy and loss to real data (did it work with same accuracy for drum beat)? **14**
- [ ] Conclusion: **15**
  - [ ] How did the models perform with respect to accuracy.
  - [ ] Did they underfit / generalise or overfit.
  - [ ] Improvements.
- [ ] Check marking criterion for extra marks **16**
  - Does it flow?
  - Are there grammar errors?
  - Diagrams, tables and figures add to flow of report?
  - Citings and figure numbers correct?

## Day progress

_AOY (anything outstanding from yesterday)_

| Monday 01                     | Tuesday 02        | Wednesday 03                     | Thursday 04 | Friday 05         | Saturday 06 & Sunday 07 |
| ----------------------------- | ----------------- | -------------------------------- | ----------- | ----------------- | ----------------------- |
| MDEs & P(1,2) & W(0,1,4,5,6a) | P(3,4,5) & W(AOY) | MEETING & P(AOY) & W(2,6b,6c,6d) | W(7,8,9,10) | W(AOY, 3?) CRYPTO | W(11,12,13,14)          |

| Monday 08   | Tuesday 09 | Wednesday 10                 | Thursday 11 | Friday 12 | Saturday 13 & Sunday 14 |
| ----------- | ---------- | ---------------------------- | ----------- | --------- | ----------------------- |
| MDEs W(AOY) | W(15, 16)  | W(AOY) MEETING (first draft) |             |           |                         |

| Monday 15 | Tuesday 16 | Wednesday 17 | Thursday 18 | Friday 19 | Saturday 20 & Sunday 21 |
| --------- | ---------- | ------------ | ----------- | --------- | ----------------------- |
|           |            | EASTER       | W(6, 3)     | W(AOY)    | P(4,5) & W(11)          |

| Monday 22 | Tuesday 23 | Wednesday 24 | Thursday 25 | Friday 26 | Saturday 27 & Sunday 28  |
| --------- | ---------- | ------------ | ----------- | --------- | -----------------------  |
| W(15)     | W(AOY)     | W(16)        | W(AOY)      | W(AOY)    | W(AOY) Finish and Submit |

| Monday 29 | Tuesday 30 | Wednesday 01 | Thursday 02 | Friday 03 |
| --------- | ---------- | ------------ | ----------- | --------- |
|           |            | MEETING?     |             |           |

**Key dates**

- Friday 3 May deadline for entire dissertation
