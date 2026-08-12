"""
Generate the static paper reproduction gallery under ``public/reproduce``.

The gallery is built mechanically from the ``meta.yaml`` files of git-tracked
reproductions in ``examples/reproduce_papers``. Untracked work in progress is
never published. Validation runs first and aborts on the first batch of errors
instead of guessing, so a malformed ``meta.yaml`` fails the build rather than
silently rendering a broken card.

Output is deterministic: rerunning without data changes produces no diff.

Usage, from ``docs/source``::

    python generate_gallery.py
"""

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PAPERS_DIR = REPO_ROOT / "examples" / "reproduce_papers"
OUT_DIR = HERE / "public" / "reproduce"

GITHUB_TREE = "https://github.com/tensorcircuit/tensorcircuit-ng/tree/master"
BOT_SUFFIX = "[bot]"
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_SUFFIXES = RASTER_SUFFIXES | {".svg"}

THUMB_SIZE = (800, 500)
THUMB_PAD = 18
FIGURE_MAX_WIDTH = 1600

REQUIRED_FIELDS = [
    "title",
    "arxiv_id",
    "url",
    "year",
    "authors",
    "tags",
    "tc_features",
    "backend",
    "hardware_requirements",
    "card_title",
    "summary",
    "description",
    "outputs",
]
MAX_CARD_TITLE = 60
MAX_SUMMARY = 140


def tracked_meta_files():
    result = subprocess.run(
        ["git", "ls-files", "examples/reproduce_papers/*/meta.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(REPO_ROOT / line for line in result.stdout.split())


def tracked_paths(relative_dir):
    result = subprocess.run(
        ["git", "ls-files", relative_dir],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


_CONTRIBUTOR_CACHE = {}


def derive_contributor(folder):
    """
    Return the earliest non-bot author in the folder's git history.

    Bot-authored reproductions fall through to the first human who touched the
    folder, because the gallery credits whoever vouches for a reproduction
    rather than whatever tool typed it. Returns None when only bots appear.
    """
    relative = folder.relative_to(REPO_ROOT).as_posix()
    if relative not in _CONTRIBUTOR_CACHE:
        result = subprocess.run(
            ["git", "log", "--reverse", "--format=%an", "--", relative],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        humans = [
            name.strip()
            for name in result.stdout.splitlines()
            if name.strip() and not name.strip().endswith(BOT_SUFFIX)
        ]
        _CONTRIBUTOR_CACHE[relative] = humans[0] if humans else None
    return _CONTRIBUTOR_CACHE[relative]


def load_taxonomy():
    taxonomy = yaml.safe_load((PAPERS_DIR / "taxonomy.yaml").read_text())
    return taxonomy["tags"], taxonomy["tc_features"], set(taxonomy["backends"])


def validate(slug, meta, folder, errors, tag_vocab, feature_vocab, backends):
    def fail(message):
        errors.append(f"{slug}: {message}")

    missing = [field for field in REQUIRED_FIELDS if field not in meta]
    for field in missing:
        fail(f"missing required field '{field}'")
    if missing:
        return

    for tag in meta["tags"]:
        if tag not in tag_vocab:
            fail(f"tag '{tag}' is not in taxonomy.yaml")
    for feature in meta["tc_features"]:
        if feature not in feature_vocab:
            fail(f"tc_feature '{feature}' is not in taxonomy.yaml")
    if meta["backend"] not in backends:
        fail(f"backend '{meta['backend']}' is not in taxonomy.yaml")

    if not str(meta["url"]).startswith("https://arxiv.org/abs/"):
        fail(f"url must use the /abs/ form, got {meta['url']}")
    if len(meta["card_title"]) > MAX_CARD_TITLE:
        fail(f"card_title is {len(meta['card_title'])} chars, limit {MAX_CARD_TITLE}")
    if len(meta["summary"]) > MAX_SUMMARY:
        fail(f"summary is {len(meta['summary'])} chars, limit {MAX_SUMMARY}")

    tracked = tracked_paths(f"examples/reproduce_papers/{slug}")
    declared = set()
    images = 0
    for output in meta["outputs"]:
        for key in ("target", "path", "script"):
            if key not in output:
                fail(f"output entry missing '{key}'")
        path = output.get("path", "")
        if path.startswith("examples/"):
            fail(f"output path must be folder-relative, got {path}")
        if not (folder / path).exists():
            fail(f"declared output does not exist: {path}")
        if f"examples/reproduce_papers/{slug}/{path}" not in tracked:
            fail(f"declared output is not git-tracked: {path}")
        declared.add(path)
        if Path(path).suffix.lower() in IMAGE_SUFFIXES:
            images += 1
        script = output.get("script", "")
        if not (folder / script).exists():
            fail(f"declared script does not exist: {script}")

    if images == 0:
        fail("at least one declared output must be an image for the gallery card")

    for image in sorted((folder / "outputs").glob("*")):
        relative = f"outputs/{image.name}"
        if image.suffix.lower() in IMAGE_SUFFIXES and relative not in declared:
            fail(f"image exists on disk but is not declared: {relative}")

    thumbnail = meta.get("thumbnail")
    if thumbnail is not None and thumbnail not in declared:
        fail(f"thumbnail '{thumbnail}' is not among the declared outputs")

    if not meta.get("contributor") and derive_contributor(folder) is None:
        fail(
            "git history has only bot authors, so the reproduction has no one "
            "vouching for it; add an explicit 'contributor' to meta.yaml"
        )


def flatten_white(image):
    image = image.convert("RGBA")
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(image, mask=image.split()[3])
    return canvas


def write_thumbnail(source, target):
    image = Image.open(source).convert("RGBA")
    image.thumbnail(
        (THUMB_SIZE[0] - 2 * THUMB_PAD, THUMB_SIZE[1] - 2 * THUMB_PAD), Image.LANCZOS
    )
    canvas = Image.new("RGB", THUMB_SIZE, "white")
    offset = ((THUMB_SIZE[0] - image.width) // 2, (THUMB_SIZE[1] - image.height) // 2)
    canvas.paste(image, offset, mask=image.split()[3])
    canvas.save(target, "WEBP", quality=88, method=6)


def write_figure(source, target):
    image = Image.open(source)
    if image.width > FIGURE_MAX_WIDTH:
        height = round(image.height * FIGURE_MAX_WIDTH / image.width)
        image = image.resize((FIGURE_MAX_WIDTH, height), Image.LANCZOS)
    flatten_white(image).save(target, "WEBP", quality=85, method=6)


def build_assets(slug, folder, meta):
    figures = []
    for output in meta["outputs"]:
        path = output["path"]
        suffix = Path(path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        stem = f"{slug}__{Path(path).stem}"
        if suffix == ".svg":
            asset = f"figures/{stem}.svg"
            shutil.copyfile(folder / path, OUT_DIR / asset)
            thumb = asset
        else:
            asset = f"figures/{stem}.webp"
            thumb = f"thumbs/{stem}.webp"
            write_figure(folder / path, OUT_DIR / asset)
            write_thumbnail(folder / path, OUT_DIR / thumb)
        figures.append(
            {
                "target": output["target"],
                "source": path,
                "figure": asset,
                "thumb": thumb,
                "script": output["script"],
            }
        )
    return figures


def report_untracked(tracked):
    """
    Warn about reproductions that exist on disk but are invisible to the gallery.

    Untracked folders are skipped on purpose so that work in progress is never
    published, but a forgotten ``git add`` looks exactly the same. Naming the
    skipped folders keeps the two cases distinguishable.
    """
    published = {meta_file.parent for meta_file in tracked}
    for folder in sorted(PAPERS_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith((".", "__")):
            continue
        if folder in published:
            continue
        reason = (
            "it is not git-tracked"
            if (folder / "meta.yaml").exists()
            else "it has no meta.yaml"
        )
        print(f"note: skipping {folder.name}, {reason}", file=sys.stderr)


def collect():
    tag_vocab, feature_vocab, backends = load_taxonomy()
    errors = []
    entries = []

    report_untracked(tracked_meta_files())

    for meta_file in tracked_meta_files():
        folder = meta_file.parent
        slug = folder.name
        meta = yaml.safe_load(meta_file.read_text())
        validate(slug, meta, folder, errors, tag_vocab, feature_vocab, backends)
        if errors:
            continue
        entries.append((slug, folder, meta))

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

    for directory in ("figures", "thumbs"):
        shutil.rmtree(OUT_DIR / directory, ignore_errors=True)
        (OUT_DIR / directory).mkdir(parents=True, exist_ok=True)

    records = []
    for slug, folder, meta in entries:
        figures = build_assets(slug, folder, meta)
        preferred = meta.get("thumbnail")
        if preferred is not None:
            figures.sort(key=lambda f: f["source"] != preferred)
        artifacts = [
            output["path"]
            for output in meta["outputs"]
            if Path(output["path"]).suffix.lower() not in IMAGE_SUFFIXES
        ]
        records.append(
            {
                "slug": slug,
                "title": meta["title"],
                "card_title": meta["card_title"],
                "summary": meta["summary"],
                "description": meta["description"],
                "authors": meta["authors"],
                "contributor": meta.get("contributor") or derive_contributor(folder),
                "year": meta["year"],
                "arxiv_id": meta["arxiv_id"],
                "url": meta["url"],
                "tags": meta["tags"],
                "tc_features": meta["tc_features"],
                "backend": meta["backend"],
                "hardware": meta["hardware_requirements"],
                "figures": figures,
                "artifacts": artifacts,
                "code_url": f"{GITHUB_TREE}/examples/reproduce_papers/{slug}",
            }
        )

    records.sort(key=lambda r: (-r["year"], r["title"].lower()))
    return records, tag_vocab, feature_vocab


PAGE_CSS = """
:root {
    --bg-primary: #04060d;
    --bg-secondary: #090c16;
    --bg-card: rgba(13, 17, 33, 0.7);
    --accent-cyan: #00f0ff;
    --accent-violet: #8a2be2;
    --text-primary: #ffffff;
    --text-secondary: #b4c3df;
    --text-muted: #6b7c96;
    --border-glass: rgba(255, 255, 255, 0.07);
    --font-sans: 'Inter', sans-serif;
    --font-display: 'Outfit', sans-serif;
    --font-mono: 'Fira Code', monospace;
    --transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-sans);
    line-height: 1.6;
    overflow-x: hidden;
}
a { color: var(--accent-cyan); text-decoration: none; }
a:hover { text-decoration: underline; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--accent-cyan), var(--accent-violet));
    border-radius: 4px;
}
.container { width: 100%; max-width: 1320px; margin: 0 auto; padding: 0 2rem; }
.gradient-text {
    background: linear-gradient(135deg, var(--accent-cyan) 0%, #a450ff 60%, #ff007f 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

header.hero {
    position: relative;
    padding: 4.5rem 0 2.5rem;
    border-bottom: 1px solid var(--border-glass);
    background:
        radial-gradient(900px 340px at 15% -10%, rgba(0, 240, 255, 0.10), transparent 70%),
        radial-gradient(760px 300px at 85% -20%, rgba(138, 43, 226, 0.14), transparent 70%);
}
.eyebrow {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.9rem;
}
h1 { font-family: var(--font-display); font-size: clamp(2.1rem, 5vw, 3.3rem); line-height: 1.15; }
.lede { color: var(--text-secondary); max-width: 62ch; margin-top: 1rem; font-size: 1.05rem; }
.disclaimer {
    margin-top: 1.6rem;
    padding: 0.95rem 1.15rem;
    border: 1px solid rgba(255, 176, 32, 0.28);
    border-left: 3px solid rgba(255, 176, 32, 0.85);
    border-radius: 10px;
    background: rgba(255, 176, 32, 0.05);
    color: var(--text-secondary);
    font-size: 0.9rem;
    max-width: 82ch;
}
.disclaimer strong { color: #ffcf72; font-weight: 600; }

.controls { padding: 2rem 0 0.5rem; }
.searchrow { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
.search {
    flex: 1 1 320px;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 0.65rem 0.95rem;
    transition: var(--transition);
}
.search:focus-within { border-color: rgba(0, 240, 255, 0.45); box-shadow: 0 0 18px rgba(0, 240, 255, 0.12); }
.search svg { flex: none; opacity: 0.55; }
.search input {
    flex: 1;
    background: none;
    border: none;
    outline: none;
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 0.95rem;
}
.search input::placeholder { color: var(--text-muted); }
select {
    background: var(--bg-secondary);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    color: var(--text-secondary);
    padding: 0.7rem 0.9rem;
    font-family: var(--font-sans);
    font-size: 0.9rem;
    outline: none;
    cursor: pointer;
}
.facet { margin-top: 1.2rem; display: flex; gap: 0.55rem; align-items: baseline; flex-wrap: wrap; }
.facet-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    min-width: 5.6rem;
}
.chip {
    border: 1px solid var(--border-glass);
    background: rgba(255, 255, 255, 0.03);
    color: var(--text-secondary);
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    cursor: pointer;
    transition: var(--transition);
    font-family: var(--font-sans);
}
.chip:hover { border-color: rgba(0, 240, 255, 0.35); color: var(--text-primary); }
.chip[aria-pressed="true"] {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.18), rgba(138, 43, 226, 0.22));
    border-color: rgba(0, 240, 255, 0.55);
    color: #fff;
}
.chip .count { opacity: 0.5; margin-left: 0.35rem; font-size: 0.76rem; }
.statusbar {
    margin: 1.6rem 0 0.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    color: var(--text-muted);
    font-size: 0.86rem;
    font-family: var(--font-mono);
}
.linkbtn {
    background: none;
    border: none;
    color: var(--accent-cyan);
    cursor: pointer;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    padding: 0;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
    gap: 1.5rem;
    padding: 1.2rem 0 5rem;
}
.card {
    position: relative;
    display: flex;
    flex-direction: column;
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    overflow: hidden;
    cursor: pointer;
    transition: var(--transition);
    text-align: left;
    font: inherit;
    color: inherit;
    padding: 0;
}
.card[hidden] { display: none; }
.card:hover, .card:focus-visible {
    transform: translateY(-4px);
    border-color: rgba(0, 240, 255, 0.4);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5), 0 0 22px rgba(0, 240, 255, 0.12);
    outline: none;
}
.card .shot {
    background: #ffffff;
    aspect-ratio: 8 / 5;
    display: block;
    width: 100%;
    object-fit: contain;
    border-bottom: 1px solid var(--border-glass);
}
.card .body { padding: 1.15rem 1.25rem 1.35rem; display: flex; flex-direction: column; gap: 0.55rem; flex: 1; }
.card h3 { font-family: var(--font-display); font-size: 1.03rem; line-height: 1.35; font-weight: 600; }
.card .paper { color: var(--text-muted); font-size: 0.8rem; line-height: 1.45; }
.card .summary { color: var(--text-secondary); font-size: 0.88rem; flex: 1; }
.card .meta {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
}
.card .meta span { border: 1px solid var(--border-glass); border-radius: 6px; padding: 0.1rem 0.45rem; }
.taglist { display: flex; gap: 0.35rem; flex-wrap: wrap; }
.taglist em {
    font-style: normal;
    font-size: 0.72rem;
    color: #9fd8ff;
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.16);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
}
.empty { padding: 4rem 0; text-align: center; color: var(--text-muted); }

.modal {
    position: fixed;
    inset: 0;
    background: rgba(2, 4, 10, 0.82);
    backdrop-filter: blur(8px);
    display: none;
    z-index: 50;
    padding: 3vh 1rem;
    overflow-y: auto;
}
.modal.open { display: block; }
.sheet {
    max-width: 980px;
    margin: 0 auto;
    background: var(--bg-secondary);
    border: 1px solid rgba(0, 240, 255, 0.18);
    border-radius: 18px;
    padding: 2rem 2.2rem 2.4rem;
}
.sheet h2 { font-family: var(--font-display); font-size: 1.6rem; line-height: 1.25; padding-right: 2.5rem; }
.sheet .authors { color: var(--text-muted); font-size: 0.86rem; margin-top: 0.6rem; }
.sheet .credit { color: var(--text-secondary); font-size: 0.84rem; margin-top: 0.45rem; }
.sheet .credit span {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-right: 0.5rem;
}
.closebtn {
    float: right;
    background: none;
    border: 1px solid var(--border-glass);
    color: var(--text-secondary);
    border-radius: 8px;
    width: 2.1rem;
    height: 2.1rem;
    cursor: pointer;
    font-size: 1.1rem;
    line-height: 1;
}
.closebtn:hover { color: #fff; border-color: rgba(0, 240, 255, 0.5); }
.sheet .actions { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 1.3rem 0; }
.btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.55rem 1.1rem;
    border-radius: 8px;
    font-family: var(--font-display);
    font-size: 0.88rem;
    font-weight: 600;
    border: 1px solid var(--border-glass);
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.04);
    transition: var(--transition);
}
.btn:hover { text-decoration: none; border-color: rgba(0, 240, 255, 0.5); }
.btn.primary { background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(138, 43, 226, 0.28)); }
.sheet h4 {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.8rem 0 0.7rem;
}
.sheet p.prose { color: var(--text-secondary); font-size: 0.92rem; }
.figure { margin-bottom: 1.4rem; }
.figure img { width: 100%; background: #fff; border-radius: 12px; display: block; }
.figure figcaption { color: var(--text-muted); font-size: 0.82rem; margin-top: 0.5rem; }
.kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.55rem; }
.kv div {
    border: 1px solid var(--border-glass);
    border-radius: 9px;
    padding: 0.55rem 0.8rem;
    font-size: 0.84rem;
    color: var(--text-secondary);
}
.kv b { color: var(--text-primary); font-weight: 600; display: block; font-size: 0.82rem; }
.kv code { font-family: var(--font-mono); font-size: 0.76rem; color: var(--text-muted); }
pre {
    background: #05070f;
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    overflow-x: auto;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--text-secondary);
}
footer {
    border-top: 1px solid var(--border-glass);
    padding: 2.5rem 0 3.5rem;
    color: var(--text-muted);
    font-size: 0.86rem;
}
@media (max-width: 640px) {
    .container { padding: 0 1.1rem; }
    .sheet { padding: 1.5rem 1.2rem 2rem; }
    .facet-label { min-width: 100%; }
}
"""

PAGE_JS = """
const DATA = JSON.parse(document.getElementById('gallery-data').textContent);
const BY_SLUG = Object.fromEntries(DATA.map(d => [d.slug, d]));
const cards = Array.from(document.querySelectorAll('.card'));
const searchInput = document.getElementById('search');
const sortSelect = document.getElementById('sort');
const countLabel = document.getElementById('count');
const emptyState = document.getElementById('empty');
const grid = document.getElementById('grid');
const active = { tags: new Set(), features: new Set() };

function apply() {
    const query = searchInput.value.trim().toLowerCase();
    let visible = 0;
    for (const card of cards) {
        const tags = card.dataset.tags.split(' ').filter(Boolean);
        const features = card.dataset.features.split(' ').filter(Boolean);
        const matchTags = [...active.tags].every(t => tags.includes(t));
        const matchFeatures = [...active.features].every(f => features.includes(f));
        const matchQuery = !query || card.dataset.search.includes(query);
        const show = matchTags && matchFeatures && matchQuery;
        card.hidden = !show;
        if (show) visible += 1;
    }
    countLabel.textContent = visible + (visible === 1 ? ' reproduction' : ' reproductions');
    emptyState.hidden = visible !== 0;
}

function sortCards() {
    const mode = sortSelect.value;
    const ordered = [...cards].sort((a, b) => {
        if (mode === 'title') return a.dataset.title.localeCompare(b.dataset.title);
        const delta = Number(a.dataset.year) - Number(b.dataset.year);
        return mode === 'oldest' ? delta || a.dataset.title.localeCompare(b.dataset.title)
                                 : -delta || a.dataset.title.localeCompare(b.dataset.title);
    });
    ordered.forEach(card => grid.appendChild(card));
}

document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const set = active[chip.dataset.facet];
        const key = chip.dataset.value;
        if (set.has(key)) { set.delete(key); chip.setAttribute('aria-pressed', 'false'); }
        else { set.add(key); chip.setAttribute('aria-pressed', 'true'); }
        apply();
    });
});

document.getElementById('reset').addEventListener('click', () => {
    active.tags.clear();
    active.features.clear();
    document.querySelectorAll('.chip').forEach(c => c.setAttribute('aria-pressed', 'false'));
    searchInput.value = '';
    apply();
});

searchInput.addEventListener('input', apply);
sortSelect.addEventListener('change', sortCards);

const modal = document.getElementById('modal');
const sheet = document.getElementById('sheet');

function esc(value) {
    return String(value).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function open(slug) {
    const item = BY_SLUG[slug];
    const figures = item.figures.map(f =>
        `<figure class="figure"><img loading="lazy" src="${esc(f.figure)}" alt="${esc(f.target)}">` +
        `<figcaption>${esc(f.target)} &middot; <code>${esc(f.source)}</code></figcaption></figure>`).join('');
    const features = item.tc_features.map(f =>
        `<div><b>${esc(FEATURES[f].label)}</b><code>${esc(FEATURES[f].api)}</code></div>`).join('');
    const artifacts = item.artifacts.length
        ? `<h4>Other artifacts</h4><p class="prose">${item.artifacts.map(a => `<code>${esc(a)}</code>`).join(', ')}</p>`
        : '';
    const command = 'git clone https://github.com/tensorcircuit/tensorcircuit-ng.git\\n' +
        `cd tensorcircuit-ng/examples/reproduce_papers/${item.slug}\\npython ${item.figures[0].script}`;
    sheet.innerHTML =
        `<button class="closebtn" id="close" aria-label="Close">&times;</button>` +
        `<h2>${esc(item.title)}</h2>` +
        `<p class="authors">${esc(item.authors.join(', '))} &middot; ${item.year} &middot; arXiv:${esc(item.arxiv_id)}</p>` +
        `<p class="credit"><span>Reproduction by</span>${esc(item.contributor)}</p>` +
        `<div class="actions">` +
        `<a class="btn primary" href="${esc(item.url)}" target="_blank" rel="noopener">Read the paper</a>` +
        `<a class="btn" href="${esc(item.code_url)}" target="_blank" rel="noopener">View the code</a>` +
        `</div>` +
        `<h4>Reproduced results</h4>${figures}` +
        `<h4>What was done</h4><p class="prose">${esc(item.description)}</p>` +
        `<h4>TensorCircuit-NG APIs used</h4><div class="kv">${features}</div>` +
        `<h4>Environment</h4><div class="kv">` +
        `<div><b>Backend</b><code>tc.set_backend("${esc(item.backend)}")</code></div>` +
        `<div><b>GPU required</b><code>${item.hardware.gpu ? 'yes' : 'no'}</code></div>` +
        `<div><b>Minimum memory</b><code>${esc(item.hardware.min_memory)}</code></div>` +
        `</div>${artifacts}` +
        `<h4>Run it yourself</h4><pre>${esc(command)}</pre>`;
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
    document.getElementById('close').addEventListener('click', close);
}

function close() {
    modal.classList.remove('open');
    document.body.style.overflow = '';
}

cards.forEach(card => {
    card.addEventListener('click', () => open(card.dataset.slug));
    card.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(card.dataset.slug); }
    });
});
modal.addEventListener('click', event => { if (event.target === modal) close(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });

sortCards();
apply();
"""


def render_card(record):
    blob = " ".join(
        [
            record["title"],
            record["card_title"],
            record["summary"],
            record["arxiv_id"],
            record["slug"],
            record["backend"],
            " ".join(record["authors"]),
            " ".join(record["tags"]),
            " ".join(record["tc_features"]),
        ]
    ).lower()
    authors = record["authors"]
    byline = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
    tags = "".join(f"<em>{html.escape(tag)}</em>" for tag in record["tags"])
    facts = "".join(
        f"<span>{html.escape(str(value))}</span>"
        for value in (record["year"], byline, record["backend"])
    )
    thumb = record["figures"][0]["thumb"]
    return f"""      <article class="card" role="button" tabindex="0"
        data-slug="{html.escape(record['slug'])}"
        data-title="{html.escape(record['title'])}"
        data-year="{record['year']}"
        data-tags="{html.escape(' '.join(record['tags']))}"
        data-features="{html.escape(' '.join(record['tc_features']))}"
        data-search="{html.escape(blob, quote=True)}">
        <img class="shot" loading="lazy" src="{html.escape(thumb)}" alt="{html.escape(record['card_title'])}">
        <div class="body">
          <h3>{html.escape(record['card_title'])}</h3>
          <p class="paper">{html.escape(record['title'])}</p>
          <p class="summary">{html.escape(record['summary'])}</p>
          <div class="meta">{facts}</div>
          <div class="taglist">{tags}</div>
        </div>
      </article>
"""


def render_chips(facet, vocab, records, key):
    counts = {}
    for record in records:
        for value in record[key]:
            counts[value] = counts.get(value, 0) + 1
    chips = []
    for value in sorted(counts, key=lambda v: (-counts[v], vocab[v]["label"])):
        chips.append(
            f'<button class="chip" type="button" data-facet="{facet}" '
            f'data-value="{html.escape(value)}" aria-pressed="false">'
            f'{html.escape(vocab[value]["label"])}<span class="count">{counts[value]}</span></button>'
        )
    return "\n          ".join(chips)


def render_page(records, tag_vocab, feature_vocab):
    features_js = {
        name: {"label": spec["label"], "api": spec["api"]}
        for name, spec in feature_vocab.items()
    }
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Executable Quantum Research Hub | TensorCircuit-NG</title>
<meta name="description" content="A curated hub of quantum papers translated into runnable, metadata-rich, and independently inspectable TensorCircuit-NG research artifacts.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{PAGE_CSS}</style>
</head>
<body>

<header class="hero">
  <div class="container">
    <p class="eyebrow">Executable Research Hub</p>
    <h1>Quantum Research, <span class="gradient-text">Made Executable</span></h1>
    <p class="lede"><strong>From published quantum knowledge to executable research infrastructure.</strong> This curated hub packages published methods as runnable, metadata-rich, and independently inspectable TensorCircuit-NG artifacts. Read the paper, inspect the implementation, and run the experiment yourself.</p>
    <p class="disclaimer"><strong>Please read:</strong> these are independent reimplementations by the TensorCircuit-NG project, not author-endorsed replications. Most are deliberately scaled down &mdash; fewer qubits, smaller bond dimensions, shorter training &mdash; so that they finish quickly on a single machine, and some make explicit modeling simplifications. Each entry documents what was changed. Treat them as executable illustrations of the physics, not as verification of the original results.</p>
  </div>
</header>

<section class="controls">
  <div class="container">
    <div class="searchrow">
      <label class="search">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="search" type="search" placeholder="Search titles, authors, methods, arXiv id" autocomplete="off">
      </label>
      <select id="sort" aria-label="Sort order">
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="title">Title A&ndash;Z</option>
      </select>
    </div>
    <div class="facet">
      <span class="facet-label">Topic</span>
      {render_chips("tags", tag_vocab, records, "tags")}
    </div>
    <div class="facet">
      <span class="facet-label">TC feature</span>
      {render_chips("features", feature_vocab, records, "tc_features")}
    </div>
    <div class="statusbar">
      <span id="count">{len(records)} reproductions</span>
      <button class="linkbtn" id="reset" type="button">reset filters</button>
    </div>
  </div>
</section>

<main class="container">
  <div class="grid" id="grid">
{"".join(render_card(record) for record in records)}  </div>
  <p class="empty" id="empty" hidden>No reproduction matches these filters.</p>
</main>

<footer>
  <div class="container">
    <p>This Hub is the literature-to-artifact layer of TensorCircuit-NG's agentic research stack. Want to add one? Pick a paper, follow
    <a href="{GITHUB_TREE}/examples/reproduce_papers">examples/reproduce_papers</a>,
    and open a pull request.
    &middot; <a href="../index.html">Documentation</a>
    &middot; <a href="../platform/index.html">Platform</a>
    &middot; <a href="../agent_landing/index.html">Agentic Research</a>
    &middot; <a href="https://github.com/tensorcircuit/tensorcircuit-ng">GitHub</a></p>
  </div>
</footer>

<div class="modal" id="modal" role="dialog" aria-modal="true"><div class="sheet" id="sheet"></div></div>

<script id="gallery-data" type="application/json">{json.dumps(records, ensure_ascii=False)}</script>
<script>const FEATURES = {json.dumps(features_js, ensure_ascii=False)};{PAGE_JS}</script>
</body>
</html>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, tag_vocab, feature_vocab = collect()
    (OUT_DIR / "data.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "index.html").write_text(
        render_page(records, tag_vocab, feature_vocab), encoding="utf-8"
    )
    figures = sum(len(record["figures"]) for record in records)
    print(f"wrote {len(records)} reproductions and {figures} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
