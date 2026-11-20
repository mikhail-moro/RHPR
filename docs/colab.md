# Running RHPR on Google Colab

The official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
uses the compute units of the signed-in Google account. The workflow checks
access first, stages Kaggle inputs on a fresh Colab VM, retrieves the run
directory, and stops the VM. The **usage** task only checks authentication.

## GitHub setup

1. Locally, sign in with the Google account that owns the Colab subscription.
   The scope list follows the Colab CLI's setup for user ADC.

   ```bash
   gcloud auth application-default login \
     --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
   ```

2. In **Settings → Secrets and variables → Actions → Secrets**, create
   **`COLAB_GOOGLE_ADC`** with the complete contents of
   `~/.config/gcloud/application_default_credentials.json`. A Gemini or Google
   Cloud API key is a different credential. The existing **`KAGGLE_API_TOKEN`**
   is used on the GitHub runner to fetch the private Kaggle inputs.

3. Generate an [age](https://github.com/FiloSottile/age) key locally. Keep
   `rhpr-colab.agekey` on your device and place only the `age1…` public
   recipient in **Settings → Secrets and variables → Actions → Variables** under
   **`COLAB_ARTIFACT_RECIPIENT`**. The GitHub repository is public. The workflow
   publishes only encrypted archives and prints deployment phase markers.

   ```bash
   age-keygen -o rhpr-colab.agekey
   age-keygen -y rhpr-colab.agekey
   ```

## Launch

Open **Actions → Colab experiment → Run workflow** after the workflow is on
`main`. Choose **usage** for the initial credential check, **cpu-smoke** to
verify deployment and result export, or **experiment** for an ML1M Last-8
configuration. The workflow loads the experiment code from `code_ref`.
Pass a pushed commit SHA to pin a run. Allow time for setup and the sequential
trials inside the remote timeout.

For example, to replay the existing Thinned64 validation diagnostic on T4
after the smoke checks:

```bash
gh workflow run colab-experiment.yml \
  -f task=experiment \
  -f code_ref=codex/kindle-mixed-views \
  -f experiment=experiments/ml1m_thinned64_full_validation.json \
  -f gpu=T4 -f timeout_minutes=30
```

The launcher supports the existing ML1M Last-8 window, pair and input-ablation
runners. Colab uses one GPU and runs their seeds and variants sequentially.
The launcher keeps the experiment's data, precision, compilation and validation
settings and verifies pinned parent output bytes before allocating a GPU.
GPU allocation retries transient HTTP 502, 503 and 504 responses and read
timeouts up to ten times within 45 minutes. The first four delays are 1, 2, 3
and 4 minutes, followed by six delays of 5 minutes. The retries reuse the same
Colab notebook ID and leave the staged experiment in place.
For Kaggle sources referenced by unversioned slug, the current output must
match the pin. The runtime records the experiment revision, input config hashes,
GPU and CUDA versions.

Use T4 when comparing timings with past T4 runs. L4, A100 and H100 are available
for new comparisons if Colab has capacity. Run both the control and the proposed
method on the same GPU type and software stack to compare wall time, and check
whether any numerical differences affect rankings.

## Results

Download the `colab-results-…` artifact and unzip it. Decrypt the result archive
and the launcher log locally.

```bash
mkdir colab-results
age -d -i rhpr-colab.agekey results.tar.age | tar -xf - -C colab-results
age -d -i rhpr-colab.agekey deployment.log.age
```

The archive includes RHPR's usual `metrics.json`, `summary.md`, run manifest,
checkpoints and prediction files when produced by the experiment, plus
`colab-run.log`. If training exits with an error and the runtime is reachable,
the workflow still retrieves the run directory before releasing it. A forced
GitHub cancellation or loss of the Colab VM can interrupt retrieval. If that
happens, check **Manage sessions** in Colab for the session name printed by
the job.

For a depth pilot interrupted after a validation, decrypt only
`checkpoint.tar.age` and upload the unchanged tar bytes as `checkpoint.bin`
in your private Kaggle dataset so Kaggle preserves the archive as a single file.
Launch the same `experiment` with its `#<trial_id>` suffix and GPU type. Set
`resume_dataset` to `owner/slug` and `resume_sha256` to the archive digest.
The job verifies the digest, restores the last epoch and its optimizer, scaler,
scheduler and RNG state, and retains the prior best-validation checkpoint.
