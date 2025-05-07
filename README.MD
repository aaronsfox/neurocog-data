This repository holds the data and code related to the work:

Rolley TL, Saunders N, Goodwill A, Major B, Shultz S, Fox AS (2025). *The Relationship Between Cognitive Performance on Generic Cognitive Tests and a Sport-Specific Decision-Making Test*.

Please cite this work accordingly if using data or code from this repository.

If you have any questions regarding this work, please contact Aaron Fox via e-mail (aaron.f@deakin.edu.au).

## code

The `code` folder contains the Python script used to read in, describe and analyse the neurocognitive performance test data. Running this code will replicate the figures and analyses included in the aforementioned work.

The Python package dependencies (and the specific versions used) associated with this code are:

- `matplotlib v3.8.0`
- `re v2.2.1`
- `numpy v1.23.5`
- `pandas v2.2.3`
- `seaborn v0.13.2`
- `scipy v1.12.0`

## data

The `data` folder contains the average performance results on the neurocognitive tests for each participant. The tests included in the dataset are:

- **n-Back Tests (1B, 2B, 3B):** Testing cognitive domain of working memory. Using a standardised deck of cards, the participant is required to remember if the current card being shown is that which was displayed n-cards prior to the current card being shown. The One-Back test required participants to remember if the card shown previously (i.e. one card ago) is the same as the card currently being displayed. The two-and three-back tests therefore extended this process to identifying if the card being shown is the same of that presented two and three cards prior. Participants undertook a one-minute familiarisation trial prior to completing a two-minute block in which data was collected.
- **One-Card Learning Test (OCL):** Testing cognitive domain of new learning. Using a standardised deck of cards the participant was required to remember if the current card being shown had been displayed previously. Participants undertook a one-minute familiarisation trial prior to completing a four-minute block in which data was collected.
- **Deary Liewald Test (SRT, 4Choice):** Testing cognitive domain of simple and choice-reaction time. To assess simple reaction time, participants were instructed to hit the ‘X’ key as soon as a cross appeared in the single box displayed. Participants undertook a familiarisation trial of eight trials. Data was then collected from 20 trials. To assess choice reaction time, participants were instructed to hit the appropriate key that correctly corresponded to the box in which a cross was displayed. Participants undertook a familiarisation trial of eight trials. Data was collected from 40 trials.
- **Sport-Specific Decision-Making Test (SportRT):** Test of sport-specific reaction time and decision making. The participant was seated at a desk in-front of a laptop with their left index finger resting on the ‘Z’ key and their right index finger resting on the ‘M’ key.  Participants were instructed to respond as quickly and as accurately as possible to a life-sized video projection of an opposition player changing direction. The participant was required to watch the stimulus video and determine which direction they thought the opposition player was going to move towards by hitting either the ‘Z’ (i.e. moving to the left-hand side of the screen) or ‘M’ key (i.e. moving to the right-hand side of the screen). A block study design was followed with participants completing ten 20-second stimulus blocks (i.e. ten two-second videos per block) with 20-seconds rest between each block. Each block consisted of randomised stimulus videos.

Each participant folder includes a file (`{partipant_id}_neurocog_data.csv`) containing their averaged results for a set of performance variables across each test. The performance variables include:

- **avgReactionTime:** Average reaction time (in milliseconds) across <u>all responses</u> during the test.
- **avgCorrectReactionTime:** Average reaction time (in milliseconds) across <u>correctly selected responses</u> during the test.
- **avgIncorrectReactionTime:** Average reaction time (in milliseconds) across <u>incorrectly selected responses</u> during the test.
- **responseAccuracy:** Percentage of correctly selected responses during the test.

## results

A folder containing the `figures` presented with this work. The figures in this folder can be reproduced via the above-mentioned Python code.