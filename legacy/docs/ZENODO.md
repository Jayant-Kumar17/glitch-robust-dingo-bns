# Archiving a release on Zenodo

Zenodo mints a DOI from a GitHub release once the repository is connected.
This cannot be completed from the command line without the repository owner's
Zenodo/GitHub OAuth grant.

1. Sign in to <https://zenodo.org> with the GitHub account that owns
   `Jayant-Kumar17/glitch-robust-dingo-bns`.
2. Open <https://zenodo.org/account/settings/github/>, refresh the repository
   list, and flip the switch for `glitch-robust-dingo-bns` to **ON**.
3. Publish (or re-publish) a GitHub release for the tag that should be archived
   (currently `v1.1.1`). Zenodo archives the release tarball and
   assigns a **version DOI** plus a **concept DOI**.
   The release should carry the detector checkpoint
   `checkpoints/glitch_detector_v1/best_glitch_detector.pt` as an asset; note
   that Zenodo archives the *source tarball*, which already contains the
   checkpoint because it is tracked in git (~566 KB, not LFS).
4. On the new Zenodo record, check that the metadata imported from
   `CITATION.cff` (title, author, licence, keywords) is correct, add the
   DINGO-BNS and Gravity Spy references as *related identifiers* (`cites`),
   and publish.
5. Paste the version DOI `10.5281/zenodo.<record>` back into:
   - `CITATION.cff`: `doi:` plus an `identifiers` entry for the concept DOI;
   - `README.md`: the DOI badge under the title and the concept DOI in
     *Citation*;
   - `paper/main.tex` Data availability: replace
     `Zenodo DOI to be added on acceptance` with
     `\url{https://doi.org/10.5281/zenodo.<record>}`.
6. Commit those edits as a patch release (`v1.1.2`) if the DOI must appear in
   the archived source; otherwise leave them on `main`.

Re-running step 3 for any later tag creates a new version under the same
concept DOI.
