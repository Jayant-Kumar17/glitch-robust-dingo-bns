# Archiving a release on Zenodo (manual steps)

No Zenodo deposit is created automatically. To mint a DOI for a tagged release:

1. Sign in to <https://zenodo.org> with the GitHub account that owns
   `Jayant-Kumar17/glitch-robust-dingo-bns`.
2. Open <https://zenodo.org/account/settings/github/>, refresh the repository
   list, and flip the switch for `glitch-robust-dingo-bns` to **ON**.
3. Publish a GitHub release for the tag (e.g. `v1.0.0`). Zenodo archives the
   release tarball automatically and assigns a version DOI plus a concept DOI.
   The release should carry the detector checkpoint
   `checkpoints/glitch_detector_v1/best_glitch_detector.pt` as an asset; note
   that Zenodo archives the *source tarball*, which already contains the
   checkpoint because it is tracked in git.
4. On the new Zenodo record, check that the metadata imported from
   `CITATION.cff` (title, author, licence, keywords) is correct, add the
   DINGO-BNS and Gravity Spy references as *related identifiers* (`cites`),
   and publish.
5. Paste the DOIs back into the repository:
   - `CITATION.cff`: add `doi: 10.5281/zenodo.<record>` (version DOI) and,
     under `identifiers`, the concept DOI.
   - `README.md`: add the DOI badge
     `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.<record>.svg)](https://doi.org/10.5281/zenodo.<record>)`
     under the title and the concept DOI in the *Citation* section.
   - the manuscript's *Data and code availability* statement.
6. Commit those edits as a patch release (`v1.0.1`) if the DOI must appear in
   the archived source; otherwise leave them on `main`.

Re-running step 3 for any later tag creates a new version under the same
concept DOI.
