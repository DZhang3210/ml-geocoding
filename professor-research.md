# Professor research — Step 10 targets (researched 2026-11-11)

Purpose: inform Step D (paper focus) and Step 10 (outreach). Sources were read on 2026-11-11; every claim links to its source. Paper lists come from each lab's own publication page unless noted.

**Main finding.** Neither target has a 2023–2026 paper on address matching, entity resolution, or POI conflation. The project's way in is therefore an *adjacent* theme each lab is funded or publishing on:
- **Mai:** location encoders and geographic bias.
- **Zhu:** spatial heterogeneity and uncertainty in data fusion.

The closest direct-topic researcher is Yingjie Hu (Buffalo), whose group wrote the POI-matching systematic review.

---

## 1. Gengchen Mai — UT Austin

| Item | Detail | Source |
|---|---|---|
| Position | Tenure-track Assistant Professor, Dept. of Geography and the Environment, UT Austin (since 2024-06-01); Visiting Faculty Researcher, Google Research (since 06/2025) | [homepage](https://gengchenmai.github.io/) |
| Lab | Spatially Explicit Artificial Intelligence (SEAI) Lab | [lab site](https://sites.utexas.edu/seai/), [homepage](https://gengchenmai.github.io/) |
| Background | PhD UCSB 2021 (Janowicz, STKO Lab); postdoc Stanford (Ermon); assistant professor at UGA 2022–24 | [homepage](https://gengchenmai.github.io/) |
| Themes (2023–26) | (1) location encoders / spatial representation learning; (2) geo-foundation models; (3) LLMs for geography: geographic QA, spatial RAG, agents; (4) geo-knowledge graphs (KnowWhereGraph); (5) GeoAI ethics and **geographic bias** (NSF-funded) | [homepage](https://gengchenmai.github.io/), [recruit page](https://gengchenmai.github.io/html/recruit.html) |
| Takes students? | **PhD only, as stated:** "1–2 fully-funded Ph.D. student[s] in Fall 2027". Apply with a CV + ≤300-word statement. Funding comes from an NSF grant on *geographic bias and debiasing for GeoAI*, NSF fellowships on ethical GeoAI, and TAships. **No statement about master's students, RAs or interns.** | [homepage](https://gengchenmai.github.io/), [recruit page](https://gengchenmai.github.io/html/recruit.html) |
| Direct-topic work | **None** on address matching, entity resolution or POI conflation in 2023–26. Closest: (a) co-author on geo-knowledge-guided GPT for extracting location descriptions (IJGIS 2023); (b) co-author on "Geoparsing: Solved or Biased?" (AGILE 2022), a spatially explicit evaluation of geoparser bias | [IJGIS 2023](https://doi.org/10.1080/13658816.2023.2266495), [AGILE 2022](https://agile-giss.copernicus.org/articles/3/9/2022/agile-giss-3-9-2022.pdf) |

### Representative recent papers

| # | Title | Venue, year | Link | Method in one line |
|---|---|---|---|---|
| 1 | TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning | NeurIPS 2024 D&B | [arXiv 2406.15658](https://arxiv.org/abs/2406.15658) | Unified library and benchmark of 15 location encoders (lat/lon → embedding) across geo-aware prediction tasks |
| 2 | Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface | ISPRS JPRS 2023 | [arXiv 2306.17624](http://arxiv.org/abs/2306.17624) | Multi-scale sinusoidal location encoder that preserves spherical distance, unlike 2D grid encoders |
| 3 | CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations | ICML 2023 | [arXiv 2305.01118](http://arxiv.org/abs/2305.01118) | Contrastive pre-training that aligns image embeddings with location-encoder embeddings |
| 4 | GeoLLM: Extracting Geospatial Knowledge from Large Language Models | ICLR 2024 | [OpenReview](https://openreview.net/forum?id=TqL2xBwXP3) | Prompts LLMs with coordinates plus nearby-place context (from OSM) to predict regional indicators |
| 5 | On the Opportunities and Challenges of Foundation Models for GeoAI | ACM TSAS 2024 | [arXiv 2304.06798](https://arxiv.org/abs/2304.06798) | Position paper and probe of LLMs / foundation models across geospatial tasks, incl. toponym recognition |
| 6 | LocDiff: Identifying Locations on Earth by Diffusing in the Hilbert Space | NeurIPS 2025 | [arXiv 2503.18142](https://arxiv.org/abs/2503.18142) | Diffusion model in a spherical-harmonics location-embedding space for geolocalization |
| 7 | Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Geospatial Reasoning Questions | ACL 2026 Findings | [arXiv 2502.18470](https://arxiv.org/abs/2502.18470) | Retrieval mixes spatial (SQL-style) and semantic candidates, then an LLM generates the answer |
| 8 | The KnowWhereGraph: A Large-Scale Geo-Knowledge Graph … | TGIS 2026 | [arXiv 2502.13874](https://arxiv.org/abs/2502.13874) | Large geo-KG integrating many sources, so it must link entities across them |

---

## 2. Di Zhu — University of Minnesota

The roadmap's link (UW-Madison GeoDS) points to a seminar page he spoke at, not his lab. Current pages: [GeoDI Lab](https://geodi.umn.edu/), [UMN profile](https://cla.umn.edu/about/directory/profile/dizhu), [personal site](https://www.dizhu-gis.com/).

| Item | Detail | Source |
|---|---|---|
| Position | Assistant Professor of GIScience, Dept. of Geography, Environment & Society, UMN Twin Cities; McKnight Land-Grant Professor 2026–28 | [UMN profile](https://cla.umn.edu/about/directory/profile/dizhu) |
| Lab | Geospatial Data Intelligence (GeoDI) Lab, director. Local host of ACM SIGSPATIAL 2025 | [lab](https://geodi.umn.edu/), [news](https://geodi.umn.edu/umn-and-geodi-lab-serve-local-hosts-acm-sigspatial-2025-minneapolis) |
| Background | PhD Peking University (Cartography & GIScience); visiting lecturer at UCL SpaceTimeLab before UMN. Start year at UMN not on the pages read | [UMN profile](https://cla.umn.edu/about/directory/profile/dizhu) |
| Themes (2023–26) | (1) GeoAI for spatial analytics: deep/graph models for **spatial regression, interpolation, downscaling**, and **spatial heterogeneity**; (2) human mobility and OD flows from mobile positioning data (social sensing); (3) spatial networks and urban complexity. The lab lists generative AI, agentic AI and geo-foundation models as tools | [research topics](https://geodi.umn.edu/research-topics), [projects](https://geodi.umn.edu/publications) |
| Takes students? | **Yes, master's included:** "looking for Master/Ph.D. Students, undergraduate students, and visiting scholars". Email a CV + at least one writing sample. The personal site says inquiries without a transcript and writing sample(s) are not considered. Current members include an MGIS student | [opportunities](https://geodi.umn.edu/opportunities), [personal site](https://www.dizhu-gis.com/), [members](https://geodi.umn.edu/members) |
| Direct-topic work | **None** on address matching, entity resolution, geocoding, POI conflation, or geographic LMs in 2023–26. Closest: (a) uncertainty in data fusion via knowledge graphs (ESWA 2026); (b) interpreting location embeddings learned from trajectories (2025); (c) home detection from mobile positioning data (SIGSPATIAL 2025) | [lab publication list](https://geodi.umn.edu/node/21) |

### Representative recent papers

| # | Title | Venue, year | Link | Method in one line |
|---|---|---|---|---|
| 1 | Uncovering spatial process heterogeneity from graph-based deep spatial regression | ISPRS JPRS 2026 | [DOI](https://doi.org/10.1016/j.isprsjprs.2025.12.008) | Graph neural regression that recovers spatially varying relationships (a GWR-style idea in a GNN) |
| 2 | RegionGCN: Spatial-Heterogeneity-Aware Graph Convolutional Networks | Annals of the AAG 2025 | [DOI](https://doi.org/10.1080/24694452.2025.2558661) | GCN whose parameters vary by region to model spatial heterogeneity |
| 3 | Spatially explicit knowledge in geo-embeddings: Interpreting location representation derived from human movement trajectories | Trans. Urban Data, Sci. & Tech. 2025 | [DOI](https://doi.org/10.1177/27541231251333348) | Probes what spatial knowledge location embeddings trained on trajectories encode |
| 4 | Modelling uncertainty in data fusion: A knowledge graph approach | Expert Systems with Applications 2026 | [DOI](https://doi.org/10.1016/j.eswa.2026.131550) | Represents and propagates uncertainty when fusing multi-source records in a KG |
| 5 | A Behavior-Informed and Geo-Context-Aware Home Detection Framework for Mobile Phone Positioning Data | ACM SIGSPATIAL 2025 | [ACM DL](https://dl.acm.org/doi/10.1145/3748636.3764171) | Infers home locations from positioning traces using behavioural and geographic-context features |
| 6 | A Graph-based Deep Population Downscaling Model on Irregular Spatial Units | GeoAI'25 (SIGSPATIAL wksp) | [DOI](https://doi.org/10.1145/3764912.3770815) | GNN that downscales population counts onto irregular polygons |
| 7 | A Gravity-informed spatiotemporal transformer for human activity intensity prediction | IEEE TPAMI 2025 | [DOI](https://doi.org/10.1109/TPAMI.2025.3625859) | Transformer with a gravity-model prior for predicting activity intensity |
| 8 | Inferring human movements after snowfall: A weather-informed graph learning model for flow redistribution | IJGIS 2025 | [DOI](https://doi.org/10.1080/13658816.2025.2524394) | Graph learning on mobility networks conditioned on weather |

---

## 3. How the project could connect

Framings are scored by fit with each lab's published themes. ✔✔ = strong (matches a funded or recurring theme), ✔ = plausible, — = weak.

| Framing of this project | Mai (SEAI) | Zhu (GeoDI) | Why |
|---|---|---|---|
| **Geographic bias / spatial heterogeneity of matching errors**: error rates by ZIP, urban vs rural, CT region | ✔✔ | ✔✔ | Mai's NSF grant is on geographic bias in GeoAI ([recruit](https://gengchenmai.github.io/html/recruit.html)); his group did a spatially explicit bias evaluation of geoparsers ([AGILE 2022](https://agile-giss.copernicus.org/articles/3/9/2022/agile-giss-3-9-2022.pdf)). Zhu's core theme is spatial heterogeneity ([RegionGCN](https://doi.org/10.1080/24694452.2025.2558661)). The test split is already by ZIP, so this costs only analysis, not new models |
| **Location encoders / spatial features**: add an encoded lat/lon of each side to the text matcher | ✔✔ | ✔ | Directly uses Mai's tools ([TorchSpatial](https://arxiv.org/abs/2406.15658), [Sphere2Vec](http://arxiv.org/abs/2306.17624)). Overture and Foursquare records carry coordinates. Risk: coordinates may make the text task trivial; design to show *where* text fails and space helps (e.g. missing house numbers) |
| **LLM / geo-foundation-model baseline** for address matching (zero-shot vs fine-tuned DistilBERT) | ✔✔ | ✔ | Mai's current direction ([GeoLLM](https://openreview.net/forum?id=TqL2xBwXP3), [foundation-model survey](https://arxiv.org/abs/2304.06798)); Zhu lists geo-foundation models as a tool ([topics](https://geodi.umn.edu/research-topics)). Compute-bounded on the free tier |
| **Retrieval with ZIP blocking** (bi-encoder recall@k → cross-encoder rerank) | ✔ | — | Matches the shape of [Spatial-RAG](https://arxiv.org/abs/2502.18470)'s spatial + semantic candidate retrieval. Currently out of Step 5's scope |
| **Uncertainty / calibration of match scores for conflation** | ✔ | ✔✔ | Zhu: [uncertainty in data fusion](https://doi.org/10.1016/j.eswa.2026.131550), [SHAP uncertainty](https://doi.org/10.1016/j.knosys.2026.115437). Cheap: calibration curves on existing predictions |
| **House-number handling** (one side missing the number) | ✔ | — | A text-modelling finding. It becomes geographic only when paired with coordinates (row 2) or regional rates (row 1) |
| **Bi- vs cross-encoder trade-offs** | — | — | NLP/IR framing, not either lab's. Keep it as method detail rather than the headline |
| **Learned matching on real cross-source pairs (Overture ↔ Foursquare)** as a data contribution | ✔ | ✔ | Both labs consume POI and mobility data ([KnowWhereGraph](https://arxiv.org/abs/2502.13874); Zhu's [visitor-enriched census](https://doi.org/10.1038/s41597-025-05410-0)); linkage quality sits upstream of their analyses |

**Fit summary.** Mai: geographic bias + location encoders + LLM baseline, but he states PhD-only recruiting. Zhu: spatial heterogeneity + uncertainty; he explicitly takes master's students.

### Candidate "specific results" for outreach

These are templates: fill the brackets from Step 6. Numbers marked *(results.md)* are final; the transformer rows are not in `results.md` yet.

1. **(Mai: bias)** "On real Overture ↔ Foursquare address pairs in Connecticut, my best matcher's recall on non-identical matches ranges from [x]% to [y]% across ZIP codes, and the gap tracks [address-format feature, e.g. share of records missing a house number] rather than [urban/rural]. Would your geographic-bias framework treat this as data bias or model bias?"
2. **(Zhu: heterogeneity / uncertainty)** "A fine-tuned DistilBERT matcher ties ESIM at rejecting look-alike streets but fails on pairs where one side lacks a house number [rate]. These failures cluster in [ZIP type], so linkage errors feeding a mobility or POI analysis would be spatially non-random."
3. **(Either: baselines)** "Rules and learned models trade off in opposite directions: libpostal's strict rule rejects 97.4% of look-alike street swaps but finds only 52.3% of non-identical true matches, while a random forest on string-similarity features finds 70.1% and rejects 88.7% (AUC 0.90)." *(results.md, case2s test)*

---

## 4. Additional professors with closer work

| Who | Position / lab | Why relevant | Takes students? | Sources |
|---|---|---|---|---|
| **Yingjie Hu** | Associate Professor, Geography, University at Buffalo (adjunct in CSE); director, GeoAI and Natural Hazards Lab | **Closest direct-topic match.** His group wrote the systematic review of POI-matching methods ([Sun, Hu et al., CEUS 2023](https://arxiv.org/abs/2310.15320)). Other work: location-description extraction with geo-knowledge-guided GPT (with Mai, [IJGIS 2023](https://doi.org/10.1080/13658816.2023.2266495)), [GALLOC](https://www.tandfonline.com/doi/full/10.1080/13658816.2025.2464643) (IJGIS 2025), [TopoBERT](http://www.acsu.buffalo.edu/~yhu42/papers/2023_IJDE_TopoBERT.pdf) (IJDE 2023) | **No Fall 2027 opening** as stated on the opportunities page | [lab](https://geoai.geog.buffalo.edu/), [opportunities](https://geoai.geog.buffalo.edu/opportunities/), [homepage](https://www.acsu.buffalo.edu/~yhu42/) |
| **Daniel W. Goldberg** | Faculty, Dept. of Geography, Texas A&M; lead researcher behind the Texas A&M GeoServices geocoding and address-normalization platform | **Geocoding systems specifically.** He built a production geocoder and co-authored a 2025 systematic review of next-generation geocoding systems for the LLM era ([Yin, Goldberg et al., arXiv 2503.18888](https://arxiv.org/abs/2503.18888)). Address matching is one stage of that pipeline | Not checked (no lab recruiting page found) | [profile](https://web.star.geoservices.tamu.edu/About/People/Goldberg.aspx), [GeoServices geocoder](https://geoservices.tamu.edu/Services/Geocode/About/) |

---

### Gaps and caveats
- Mai's homepage was read verbatim only through about half its length. Papers past that point were not checked.
- Zhu's start year at UMN and his CV were not read.
- Goldberg's current rank is shown as "Assistant Professor" on an older GeoServices profile, which is likely outdated. Verify on the TAMU Geography directory before contacting him.
- The geographic-bias framing (§3 row 1) assumes the test ZIPs have enough non-identical matches per ZIP for stable rates. Check n per ZIP before committing to it.
