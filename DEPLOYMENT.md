# Deploying the app on Streamlit Community Cloud

This guide publishes the existing application (`app.py`: Word OCR, Batch, Full-page OCR) as a public web app on
[Streamlit Community Cloud](https://streamlit.io/cloud). The OCR runs inside the app on the host's CPU with the
project's own CRNNv2 + BiLSTM + CTC model. There is no API key, no external OCR or AI service, no pretrained model, and
the app downloads nothing at run time.

## What gets deployed

Community Cloud clones the GitHub repository and runs `app.py`. Only what is committed to the repository is deployed.

| Item | In the repository? | Deployed? |
|---|---|---|
| Source code, configuration, fonts, `pageocr/protected_manifest.json` (fingerprints only, no content) | yes | yes |
| Training, validation and test datasets, `real_docs/`, `development_pages/`, research reports | no (ignored by `.gitignore`) | no |
| Trained checkpoint `runs/experiment_002a/best.pth` | **no**, see step 1 | only if you add it |

Files used for the deployment:

- `requirements.txt`: Python packages for the hosted app. The versions match `requirements-runtime.txt`, but torch is the
  CPU-only build, OpenCV is the headless build of the same version (no GUI libraries), and `editdistance` is added for
  the diagnostic view. Community Cloud picks this file up automatically. For a local install, keep using
  `requirements-runtime.txt` (docs/INSTALLATION.md).
- `.streamlit/config.toml`: no usage statistics, 20 MB upload limit, short error messages.
- No `packages.txt` is needed: the app needs no system packages.

## Step 1: make the checkpoint available (required for OCR)

Without a checkpoint the app still starts, but it shows **"Model not available"** and every recognition button is
disabled. It never downloads or substitutes another model.

**The problem.** `runs/experiment_002a/best.pth` is 132.6 MB (44.2 MB of model weights plus 88.3 MB of saved optimizer
state). GitHub refuses ordinary files above 100 MB, so the file cannot be committed normally.

**Recommended: Git LFS, with the file unchanged.** Community Cloud supports repositories that use Git LFS. The
registered SHA-256 (`fb29131e…62d5f1`) and the checkpoint format stay exactly as they are.

Decide these first, because they are owner decisions:

- Publishing the weights makes them public with the repository (anyone can download them). The model was trained partly
  on the IIIT-INDIC-HW-WORDS handwriting data. No licence for that dataset was found on its CVIT page or its Hugging
  Face card (checked 2026-10-11). Check that you are allowed to publish weights trained on it. To avoid publishing the weights, make the repository private (Community Cloud
  can deploy from a private repository you give it access to) or deploy without the checkpoint.
- GitHub's free Git LFS allowance is currently 10 GiB of storage and 10 GiB of download bandwidth per month. Each time
  Community Cloud builds or reboots the app it downloads the 133 MB file, which counts against the bandwidth.

Commands, run from the repository root (review before running; nothing here pushes except the last line):

```bash
git lfs install                                        # once per computer
git lfs track "runs/experiment_002a/best.pth"          # writes the rule to .gitattributes
git add .gitattributes
git add -f runs/experiment_002a/best.pth               # -f: *.pth stays ignored for every other file
git lfs ls-files                                       # must list runs/experiment_002a/best.pth
git status                                             # check that nothing else is staged
git commit -m "Add Experiment 002-A checkpoint via Git LFS for deployment"
git push origin main
```

Only the default checkpoint is needed. The other three registered checkpoints show "(file missing)" in the sidebar
and cannot be selected for OCR unless you add them the same way.

**Not recommended without a separate decision:** saving a smaller "weights only" copy of the checkpoint. Its SHA-256
would differ from the evaluated file, so it would need a new registry entry and its own provenance record. Downloading
the checkpoint at start-up from another site is also not done, by design.

## Step 2: create the app on Community Cloud

1. Go to <https://share.streamlit.io> and sign in with the GitHub account that owns the repository. Authorise
   Streamlit when asked (for a private repository it needs access to private repositories).
2. Choose **Create app** → **Deploy a public app from GitHub** (or "from your repositories").
3. Fill in:
   - Repository: `Sanjai-Rajan/Tamil-OCR-CRNN-BiLSTM-CTC`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: any free subdomain you like.
4. Open **Advanced settings**:
   - Python version: **3.12** (the version the project was tested with).
   - Secrets: leave **empty**. The app needs none.
5. Click **Deploy**. The first build installs about 46 packages (torch CPU is roughly 200 MB) and takes several
   minutes; the log is shown on the right.

## Step 3: check the running app

1. The status box at the top must say: **"Model ready: Experiment 002-A (default, frozen) on CPU. Checkpoint SHA-256
   verified."** If it says "Model not available", read the message: the file is missing (step 1) or is not the
   registered file (for example a Git LFS pointer instead of the real file: check `git lfs ls-files`).
2. Single image tab: upload a clean image of one printed Tamil word and press **Run OCR**.
3. Full-page OCR tab: upload a clean printed page and press **Read page**.
4. Wrong file types, corrupt files, files over 20 MB, and images of only two gray levels must be refused with a
   message.

## Limits of the hosted app

- **CPU only.** The device selector offers Auto and CPU. A full page of a few hundred words takes noticeably longer
  than on a GPU.
- **Resources.** Community Cloud gives every app the same limited memory and CPU; at the time of writing the
  documentation lists up to about 2.7 GB of memory and 2 CPU cores, and these limits can change. The app keeps one model
  in memory, shared by all visitors, and calls to it are serialised.
- **Sleeping.** Apps that receive no traffic go to sleep; the next visitor has to wake the app up and wait for the
  model to load again.
- **Privacy.** Visitors' images are uploaded to Streamlit's servers (hosted in the United States). They are kept in
  memory only (decoded uploads leave the cache after 15 minutes) and are not written to disk. The diagnostic view's
  "Save bundle to diagnostic_runs/" button, which writes to disk, is offered only when the app is opened on the
  computer that runs it; the ZIP download remains available. Tell visitors not to upload confidential documents
  (the About tab says so).
- **Research-data guard.** Uploads that are byte-identical to the project's protected research files are refused,
  using the fingerprints in `pageocr/protected_manifest.json`. Do not remove that file: without it the guard refuses
  every upload (fail closed).
- **Accuracy.** No accuracy on real printed pages has been measured; the app says so in the About tab. Do not
  advertise accuracy figures for the public app.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Build fails while installing packages | Open the build log. Check that Python 3.12 is selected; then Manage app → Reboot. |
| "Model not available … not found at runs/experiment_002a/best.pth" | The checkpoint is not in the repository: step 1. |
| "… is not the registered file" | The file differs from the evaluated one, often a Git LFS pointer file. Run `git lfs ls-files`, `git lfs push --all origin`. |
| App does not start, health check fails | Remove a `port` line from `.streamlit/config.toml` if one was added. If the log shows the server cannot be reached, remove the line `address = "localhost"` from `.streamlit/config.toml` (it keeps a local run private; Community Cloud expects the default). |
| "This app has gone over its resource limits" | Manage app → Reboot. Very large pages use the most memory; crop them first. |

## Running locally (unchanged)

```bash
python -m streamlit run app.py
```

See README.md and docs/INSTALLATION.md. The local app listens on `localhost` only.
