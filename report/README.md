# Report (LaTeX)

```
report/
├── main.tex              # entry point: metadata + list of chapters
├── preamble.tex          # packages, layout, macros (\todo, \env, RL notation)
├── frontmatter/          # title page
├── chapters/             # one .tex per section:
│   ├── 00_abstract  01_introduction  02_problem_formulation  03_methods
│   ├── 04_reward_design  05_experimental_protocol  06_results
│   └── 07_discussion  08_conclusion  A_appendix
├── figures/              # plots/images used by the report (PDF preferred)
├── bibliography/         # references.bib
├── .latexmkrc            # build config (output goes to build/)
├── build.ps1 / build.sh  # build scripts
└── build/                # generated, git-ignored (main.pdf is here)
```

The result of a build is always **`report/build/main.pdf`**.

## Building the PDF

You need **either** a local TeX Live (with `latexmk`) **or** Docker. Docker is the easiest option:
nothing has to be installed except Docker itself, and everyone on the team gets the same LaTeX version.

### Option 1: Docker (recommended)

**1. Prerequisites.** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and
**start it**; wait until it reports that the engine is running. Check with:

```
docker info
```

If this prints `failed to connect to the docker API ... daemon is running`, Docker Desktop is not started yet.

**2. Build with the script** (run from the `report/` folder). It uses the `texlive/texlive` image and
downloads it automatically on the first run (several GB, so the first build takes a while; later builds are fast).

```powershell
# Windows PowerShell
cd report
.\build.ps1
```

```bash
# Linux / macOS / Git Bash
cd report
./build.sh
```

If you get "running scripts is disabled" in PowerShell, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**3. Or run the container by hand** (what the scripts do internally). The current folder is mounted into the
container as `/doc`, and `latexmk` runs pdflatex and bibtex as many times as needed:

```powershell
# Windows PowerShell, from report/
docker run --rm -v "${PWD}:/doc" -w /doc texlive/texlive:latest latexmk main.tex
```

```bash
# Linux / macOS, from report/
docker run --rm -v "$PWD:/doc" -w /doc texlive/texlive:latest latexmk main.tex

# Git Bash on Windows needs MSYS_NO_PATHCONV=1 in front of docker
MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/doc" -w /doc texlive/texlive:latest latexmk main.tex
```

**4. Interactive container** (to experiment or debug a build):

```powershell
docker run -it --rm -v "${PWD}:/doc" -w /doc texlive/texlive:latest bash
# inside the container:
latexmk main.tex          # build
latexmk -C                # clean everything
exit
```

`--rm` deletes the container when it exits. Your files are not lost: they live in the mounted `report/` folder.

### Option 2: local TeX Live / MiKTeX

If `latexmk` is on your `PATH`, the scripts use it instead of Docker. You can also call it directly:

```
latexmk main.tex
```

In VS Code, the LaTeX Workshop extension picks up `.latexmkrc` automatically.

### Cleaning

```powershell
.\build.ps1 -Clean     # Windows
./build.sh clean       # Linux / macOS / Git Bash
```

This simply deletes the `build/` folder.

## Troubleshooting

| Problem | Fix |
|---|---|
| `failed to connect to the docker API` | Docker Desktop is not running. Start it and retry. |
| Build fails with a LaTeX error | Open `build/main.log` and search for `!` (error) or the `.tex:LINE:` pattern (`-file-line-error` is on). |
| `??` instead of citation or reference numbers | Normally fixed within the same `latexmk` run. If it persists, run `.\build.ps1 -Clean` and build again. |
| `Citation ... undefined` | The key is missing in `bibliography/references.bib`, or it is misspelled. |
| `File '...png' not found` | Put the figure into `figures/` and reference it by file name only, without the folder. |
| Files in `build/` belong to root (Linux) | Docker wrote them as root: `sudo chown -R "$USER" build`. |
| `Mount denied` / empty folder in the container (Windows) | Docker Desktop, Settings, Resources, File sharing: make sure the drive with the project is shared. |

## Conventions

- **One chapter = one file** in `chapters/`; each section marks the related work as `\paragraph{State of the art.}`
  or a `State of the art` subsection. Add a new chapter to the list in `main.tex`. Chapters are pulled in with
  `\input` (not `\include`), so they do not force page breaks.
- **Placeholders:** use `\todo{...}` (red, visible in the PDF). Search for `TODO` before submitting.
- **Figures:** save plots from the notebooks into `report/figures/` as PDF (vector) and reference them by
  name only (`\includegraphics{learning_curves}`, thanks to `\graphicspath`).
- **Cross-references:** label as `sec:`, `fig:`, `tab:`, `eq:` and use `\cref{...}`.
- **Citations:** add entries to `bibliography/references.bib`, cite with `\cite{key}`.
- **Commit only sources.** `build/` and all LaTeX intermediates are git-ignored; do not commit the PDF.
  Large figures can be regenerated from the notebooks, commit the final ones that the report needs.
- Edit title, authors, course and year at the top of `main.tex`.
