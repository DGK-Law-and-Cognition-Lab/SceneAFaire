# Scènes à Faire: Study 1 materials

Materials for Study 1 of the Scènes à Faire Human Validation Project (David G. Kamper, Mira Patel and Keith J. Holyoak, UCLA Department of Psychology). The preregistration itself is a single document registered on OSF. It cites the release `study1-v2-prereg` of this repository and its commit, which fix these files as of registration.

| Folder | Contents |
|---|---|
| `Experiment/Study 1 v2/` | The experiment as deployed on Pavlovia (genres-and-themes/experiment-1), with its stimulus files. The SONA credit token is blanked in this copy. |
| `Analysis/v4_rebuild/10_human_study_stimuli/` | The 336 rated items with their predictors (`stimuli_long.csv`), the frozen predictors for every eligible element (`eligible_pool_predictors.csv`), the coding vocabulary (`coding_vocabulary.csv`), the stimulus report, and the review record and rejection log. |
| `Analysis/study1_coding/` | The coder rubric, the coding module and its offline tests, and the six genre centroids used by the S1 check. |
| `Analysis/notebooks/11_study1_response_coding.ipynb` | The notebook that codes the generation responses (preregistration Section 6.4). |
| `Analysis/v4_rebuild/05_embeddings/` | The index of the frozen voyage-4-large vocabulary embeddings. The embeddings themselves (`element_embeddings.npy`, 167 MB) are attached to the release; place the file in this folder before running the notebook or the tests. |
| `Analysis/study1_power/` | The power simulation script and its output (preregistration Section 3.4). |
| `Docs/Preregistration/` | The human coding manual and the assignment schedule. |

The folder layout mirrors the project's, so the notebook's paths work unchanged when it is run from the project folder.
