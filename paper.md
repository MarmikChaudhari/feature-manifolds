[Transformer Circuits Thread](https://transformer-circuits.pub/)

# When Models Manipulate Manifolds: The Geometry of a Counting Task

# When Models Manipulate Manifolds: The Geometry of a Counting Task

![](<Base64-Image-Removed>)

### Authors

Wes Gurnee\*,Emmanuel Ameisen\*,Isaac Kauvar,Julius Tarng,Adam Pearce,Chris Olah,Joshua Batson\*‡

### Affiliations

[Anthropic](https://www.anthropic.com/)

### Published

October 21st, 2025

\\* Core Research Contributor;‡ Correspondence to [joshb@anthropic.com](mailto:joshb@anthropic.com)

### Authors

### Affiliations

### Published

_Not published yet._

### DOI

_No DOI yet._

### [Introduction](https://transformer-circuits.pub/2025/linebreaks/index.html\#introduction)

Intelligent systems need perception to understand, predict, and navigate their environment. These sensory capabilities reflect what's useful for survival in a specific environment: bats use echolocation, migratory birds sense magnetic fields, Arctic reindeer shift their UV vision seasonally. But when your world is made of text, what do you see?  Language models encounter many text-based tasks that benefit from visual or spatial reasoning: parsing ASCII art, interpreting tables, or handling text wrapping constraints. Yet their only “sensory” input is a sequence of integers representing tokens. They must learn perceptual abilities from scratch, developing specialized mechanisms in the process.

In this work, we investigate the mechanisms that enable Claude 3.5 Haiku to perform a natural perceptual task which is common in pretraining corpora and involves tracking position in a document. We find learned representations of position that are in some ways quite similar to the biological neurons found in mammals who perform analogous tasks (“place cells” and “boundary cells” in mice), but in other ways unique to the constraints of the residual stream in language models. We study these representations and find dual interpretations: we can understand them as a family of discrete features or as a one-dimensional “feature manifold”/“multidimensional feature”

- **Feature Manifold Toy Model** [\[link\]](https://transformer-circuits.pub/2023/may-update/index.html#feature-manifolds)

  C. Olah, J. Batson.

   2023\.
- **What is a Linear Representation? What is a Multidimensional Feature?** [\[link\]](https://transformer-circuits.pub/2024/july-update/index.html#linear-representations)

  C. Olah. 2024.
- **Curve Detector Manifolds in InceptionV1** [\[link\]](https://livgorton.com/curve-detector-manifolds/)

  L. Gorton. 2024.
- **Not All Language Model Features Are One-Dimensionally Linear** [\[link\]](https://openreview.net/forum?id=d63a4AM4hb)

  J. Engels, E.J. Michaud, I. Liao, W. Gurnee, M. Tegmark.

  The Thirteenth International Conference on Learning Representations. 2025.

\[1, 2, 3, 4\]

.

1

All features have a magnitude dimension; so a discrete feature is a one-dimensional ray, and a one-dimensional feature manifold is the set of all scalings of that manifold, contracting to the origin. See [What is a Linear Representation? What is a Multidimensional Feature?](https://transformer-circuits.pub/2024/july-update/index.html#linear-representations) In the first interpretation, position is determined by which features activate and how strongly; in the latter interpretation, it's determined by angular movement on the feature manifold. Similarly, computation has two dual interpretations, as discrete circuits or geometric transformations.

The task we study is linebreaking in fixed-width text. When training on source code, chat logs, email archives, scanned articles, or judicial rulings that have line width constraints, how does the model learn to predict when to break a line?

2

Michaud et al. looked for “quanta” of model skills by clustering gradients

- **The Quantization Model of Neural Scaling** [\[link\]](https://openreview.net/forum?id=3tbTw2ga8K)

  E.J. Michaud, Z. Liu, U. Girit, M. Tegmark.

  Thirty-seventh Conference on Neural Information Processing Systems. 2023.

\[5\]

. Their Figure 1 shows that predicting newlines in fixed-width text formed one of the top 400 clusters for the smallest model in the Pythia family, with 70m parameters. Human visual perception lets us do this almost completely subconsciously – when writing a birthday card, you can see when you are out of room on a line and need to begin the next – but language models see just a list of integers. In order to correctly predict the next token, in addition to selecting the next word, the model must somehow count the characters in the current line, subtract that from the line width constraint of the document, and compare the number of characters remaining to the length of the next word. As a concrete example, consider the below pair of prompts with an implicit 50-character line wrapping constraint.

3

The wrapping constraint is implicit. Each newline gives a lower bound (the previous word did fit) and an upper bound (the next word did not). We do not nail down the extent to which the model performs optimal inference with respect to those constraints, rather focusing on how it approximately uses the length of each preceding line to determine whether to break the next. There are also many edge cases for handling tokenization and punctuation. A model could even attempt to infer whether the source document used a non-monospace font and then use the pixel count rather than the character count as a predictive signal! When the next word fits, the model says it; when it does not, the model breaks the line:

![](<Base64-Image-Removed>)

To orient ourselves to the stages of the computation, we first studied the model using discrete dictionary features. In this frame, we can understand computation as an “attribution graph”

- **Circuit Tracing: Revealing Computational Graphs in Language Models** [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

  E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, N.L. Turner, B. Chen, C. Citro, D. Abrahams, S. Carter, B. Hosmer, J. Marcus, M. Sklar, A. Templeton, T. Bricken, C. McDougall, H. Cunningham, T. Henighan, A. Jermyn, A. Jones, A. Persic, Z. Qi, T. Ben Thompson, S. Zimmerman, K. Rivoire, T. Conerly, C. Olah, J. Batson.

  Transformer Circuits. 2025.

\[6\]

 where a cascade of features excite or inhibit each other.

4

We actually first tried to use patching and probing without looking at the graph as a kind of methodological test of the utility of features, but did not make much progress. In hindsight, we were training probes for quantities different than the ones the model represents cleanly, e.g., a fusion of the current token position and the line width.

![](<Base64-Image-Removed>)Attribution Graph for Claude 3.5 Haiku’s prediction of a newline in the aluminum prompt. We see features relating to “width of the previous line” and “position in the current line” which together activate features for “distance from line limit”. Combined with features for the planned next word, these features activate “predict newline” features.

The attribution graph shows how the model performs this task by combining features that represent different concepts it needs to track:

1. Features for the current position in the line (the character count) as well as features for the total line width (the constraint) are computed by accumulating features for individual token lengths.
2. The model then combines these two representations — current position and line width — to estimate the distance from the end of the line, leading to “characters remaining” features.
3. Finally, the model uses this estimate of characters remaining along with features for the planned next word to determine if the next word will fit on the line or not.

The attribution graph provides a kind of execution trace of the algorithm, showing on this prompt which variables are computed and from what. After finding large feature families involved in representing these quantities across a diverse dataset, we suspected a simpler lens might be provided in terms of lower-dimensional feature manifolds interacting geometrically. We found geometric perspectives on the following questions:

![](<Base64-Image-Removed>)Key steps in the linebreaking behavior can be described in terms of the construction and manipulation of manifolds.

How does the model represent different counts? The number of characters in a token, the number of characters in the current line, the overall line width constraint, and the number of characters remaining in the current line are each represented on 1-dimensional feature manifolds embedded with high curvature in low-dimensional subspaces of the residual stream. These manifolds have a dual interpretation in terms of discrete features, which tile the manifold in a canonical way, providing approximate local coordinates. Manifolds with similar geometry arise for a variety of ordinal concepts, and a ringing pattern we see in the embedded geometry in all these cases is optimal with respect to a simple physical model (§ [Representing Character Count](https://transformer-circuits.pub/2025/linebreaks/index.html#char-count)).

5

Ringing, in the manifold perspective, corresponds to interference in the feature superposition perspective.

How does the model detect the boundary? To detect an approaching line boundary, the model must compare two quantities: the current character count and the line width. We find attention heads whose QK matrix rotates one counting manifold to align it with the other at a specific offset, creating a large inner product when the difference of the counts falls within a target range. Multiple heads with different offsets work together to precisely estimate the characters remaining (§ [Sensing the Line Boundary](https://transformer-circuits.pub/2025/linebreaks/index.html#boundary)).

How does the model know if the next word fits? The final decision — whether to predict a newline — requires combining the estimate of characters remaining with the length of the predicted next word. We discover that the model positions these counts on near-orthogonal subspaces, creating a geometric structure where the correct linebreak prediction is linearly separable (§ [Predicting the Newline](https://transformer-circuits.pub/2025/linebreaks/index.html#prediction)).

How does the model construct these curved geometries? The curvature in the character count representation manifold is produced by many attention heads working together, each contributing a piece of the overall curvature. This distributed algorithm is necessary because individual components cannot generate sufficient output variance to create the full representation (§ [A Distributed Character Counting Algorithm](https://transformer-circuits.pub/2025/linebreaks/index.html#count-algo)).

We validate these interpretations through targeted interventions, ablations, and “visual illusions” — character sequences that hijack specific attention mechanisms to disrupt spatial perception (§ [Visual Illusions](https://transformer-circuits.pub/2025/linebreaks/index.html#illusion)).

Zooming out, we take several broader lessons from this mechanistic case study:

When Models Manipulate Manifolds. For representing a scalar quantity (e.g., integer counts from 111
1 to NNN
N), it is inefficient to use NNN
N orthogonal dimensions, and not expressive enough to use just one

6

Orthogonal dimensions would also not be robust to estimation noise.. Instead models learn to represent these quantities on a feature manifold with intrinsic dimension 1 (the count) embedded in a subspace with extrinsic dimension 1<d≪N1 < d \\ll N1<d≪N
1 < d \\ll N (e.g.,

- **Curve Detector Manifolds in InceptionV1** [\[link\]](https://livgorton.com/curve-detector-manifolds/)

  L. Gorton. 2024.
- **Not All Language Model Features Are One-Dimensionally Linear** [\[link\]](https://openreview.net/forum?id=d63a4AM4hb)

  J. Engels, E.J. Michaud, I. Liao, W. Gurnee, M. Tegmark.

  The Thirteenth International Conference on Learning Representations. 2025.
- **The Origins of Representation Manifolds in Large Language Models**

  A. Modell, P. Rubin-Delanchy, N. Whiteley.

  arXiv preprint arXiv:2505.18235. 2025.

\[3, 4, 7\]

), in which the curve “ripples”. Such rippled manifolds optimally trade off capacity constraints (roughly, dimensionality) with maintaining the distinguishability of different scalar values (curvature). Our work demonstrates the intricate ways in which these manifolds can be manipulated to perform computation and show how this can require distributing computation across multiple model components.

Duality of Features and Geometry. Dictionary features provide an unsupervised entry point for discovering mechanisms, and attribution graphs surface the important features for any particular prediction. Sometimes, discrete features (and their interactions) can be equivalently described using continuous feature manifolds (and their transformations). In cases where it is possible to explicitly parameterize the manifold (as with the various integer counts we study), we can directly study the geometry, making some operations clearer (e.g., boundary detection). But this approach is expensive in researcher time and potentially limited in scope: it's straightforward when studying known continuous variables but becomes difficult to execute correctly for more complex, difficult-to-parametrize concepts.

Complexity Tax. While unsupervised discovery is a victory in and of itself, dictionary features fragment the model into a multitude of small pieces and interactions – a kind of complexity tax on the interpretation. In cases where a manifold parametrization exists, we can think of the geometric description as reducing this tax. In other cases, we will need additional tools to reduce the interpretation burden, like hierarchical representations

- **From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit**

  V. Costa, T. Fel, E.S. Lubana, B. Tolooshams, D. Ba.

  arXiv preprint arXiv:2506.03093. 2025.

\[8\]

 or macroscopic structure in the global weights

- **Interpretability Dreams** [\[HTML\]](https://transformer-circuits.pub/2023/interpretability-dreams/index.html)

  C. Olah. 2023.

\[9\]

. We would be excited to see methods that extend the dictionary learning paradigm to unsupervised discovery of other kinds of geometric structures (e.g., those found in prior work

- **A structural probe for finding syntax in word representations** [\[PDF\]](https://aclanthology.org/N19-1419.pdf)

  J. Hewitt, C.D. Manning.

  Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4129--4138. 2019.

  [DOI: 10.18653/v1/N19-1419](https://doi.org/10.18653/v1/N19-1419)
- **Visualizing and measuring the geometry of BERT** [\[PDF\]](https://proceedings.neurips.cc/paper_files/paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf)

  A. Coenen, E. Reif, A. Yuan, B. Kim, A. Pearce, F. Viégas, M. Wattenberg.

  Advances in Neural Information Processing Systems, Vol 32. 2019.
- **The geometry of multilingual language model representations**

  T.A. Chang, Z. Tu, B.K. Bergen.

  arXiv preprint arXiv:2205.10964. 2022.
- **Relational composition in neural networks: A survey and call to action**

  M. Wattenberg, F.B. Viegas.

  arXiv preprint arXiv:2407.14662. 2024.
- **The geometry of categorical and hierarchical concepts in large language models**

  K. Park, Y.J. Choe, Y. Jiang, V. Veitch.

  arXiv preprint arXiv:2406.01506. 2024.
- **The geometry of concepts: Sparse autoencoder feature structure**

  Y. Li, E.J. Michaud, D.D. Baek, J. Engels, X. Sun, M. Tegmark.

  Entropy, Vol 27(4), pp. 344. MDPI. 2025.
- **The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence** [\[link\]](https://arxiv.org/pdf/2502.17420)

  T. Wollschlager, J. Elstner, S. Geisler, V. Cohen-Addad, S. Gunnemann, J. Gasteiger.

  arXiv preprint arXiv:2502.17420. 2025.
- **Projecting assumptions: The duality between sparse autoencoders and concept geometry**

  S.S.R. Hindupur, E.S. Lubana, T. Fel, D. Ba.

  arXiv preprint arXiv:2503.01822. 2025.
- **The Origins of Representation Manifolds in Large Language Models**

  A. Modell, P. Rubin-Delanchy, N. Whiteley.

  arXiv preprint arXiv:2505.18235. 2025.

\[10, 11, 12, 13, 14, 15, 16, 17, 7\]

).

Natural Tasks. The crispness of the representations and circuits we found was quite striking, and may be due to how well the model does the task. Linebreaking is an extremely natural behavior for a pretrained language model, and even tiny models are capable of it given enough context. Studying tasks which are natural for pretrained language models, instead of those of more theoretical interest to human investigators, may offer promising targets for finding general mechanisms.

#### Preliminaries

To enable systematic analysis, we created a synthetic dataset using a text corpus of diverse prose where we (1) stripped out all newlines and (2) reinserted newlines every kkk
k characters to the nearest word boundary ≤k\\leq k≤k
\\leq k for k=15,20,…,150k=15,20,\\ldots,150k=15,20,…,150
k=15,20,\\ldots,150. As an example, here is the opening sentence of the Gettysburg Address, wrapped to k=40k=40k=40
k=40 characters, with the newlines shown explicitly.

Four score and seven years ago our⏎

fathers brought forth on this continent,⏎

a new nation, conceived in Liberty, and⏎

dedicated to the proposition that all⏎

men are created equal.

Claude 3.5 Haiku is able to adapt to the line length for every value of kkk
k, predicting newlines at the correct positions with high probability by the third line (see [Appendix](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-task-performance)).

All features in the main text of this paper are from a 10 million feature Weakly Causal Crosscoder (WCC) dictionary

- **Sparse Crosscoders for Cross-Layer Features and Model Diffing** [\[HTML\]](https://transformer-circuits.pub/2024/crosscoders/index.html)

  J. Lindsey, A. Templeton, J. Marcus, T. Conerly, J. Batson, C. Olah.

   2024\.

\[18\]

 trained on Claude 3.5 Haiku. Feature activation values are normalized to their max throughout.

### [Representing Character Count](https://transformer-circuits.pub/2025/linebreaks/index.html\#char-count)

We define the line character count (or character count) at a given token in a prompt to be the total number of characters since the last newline, including the characters of the current token.

A natural thing to check is if the model linearly represents the character count as a quantitative variable: that is, can we predict character count with high accuracy via linear regression on the residual stream? Yes: a linear probe fit on the residual stream after layer 1 has an R2R^2R2
R^2 of 0.985. This success does not mean, however, that the model actually represents the character count along a single line.

Instead, we find a multidimensional representation of the character count that we will analyze from four perspectives:

1. Sparse crosscoder features.











   7

   Each feature has an encoder, which acts as a linear + (Jump)ReLU probe on the residual stream, and a decoder. Ten features f1,…,f10f\_1,\\ldots,f\_{10}f1​,…,f10​
   f\_1,\\ldots,f\_{10} are associated with line character count. The model's estimate of the character count, given a residual stream vector xxx
   x, is summarized by the set of activities of each of the 10 features {fi(x)}\\{f\_i(x)\\}{fi​(x)}
   \\{f\_i(x)\\}.
2. A low-dimensional subspace.











   8

   The model's estimate of the character count is summarized by the projection π(x)\\pi(x)π(x)
   \\pi(x) of xxx
   x onto that subspace. Two datapoints have similar character counts if their projections are close in that subspace.
3. A continuous 1-dimensional manifold contained in that low-dimensional subspace.











   9

   The model's estimate of the character count is summarized by the nearest point on the manifold to the projection of xxx
   x into the subspace, and its confidence in that estimate by the magnitude of π(x)\\pi(x)π(x)
   \\pi(x).
4. A set of 150 logistic probes (corresponding to values of line character count from 1 to 150).











   10

   The model's estimate of the character count is summarized by the probability distribution given by the softmax of the probe activities, softmax(Px)(Px)(Px)
   (Px).

Each of these perspectives provides a complementary view of the same underlying object. The feature perspective is valuable for getting oriented, the subspace is perfect for causal intervention, the manifold is helpful for understanding how the representation is constructed and then manipulated to detect boundaries, and the logistic probes are useful for analyzing the OV and QK matrices of the individual attention heads involved.

#### Character Count Features

We begin with the features. In layers one and two, we found features that seemed to activate based on a token’s character position within a line. For example, in the attribution graph for the aluminum prompt, there were two features active on the final word “called” that seemed to fire when the line character count was between 35–55 and 45–65, respectively. To find more such features, we computed the mean activation of each feature binned by line character count. There were ten features with smooth profiles and large between-character-count variance, shown below:

![](<Base64-Image-Removed>)A family of features representing the current character count in a line of text. The tuning curve of the features’ activity increases at larger line character counts.

We find these features especially interesting as they are quite analogous to curve-detector features in vision models

- **Curve Circuits** [\[link\]](https://distill.pub/2020/circuits/curve-circuits)

  N. Cammarata, G. Goh, S. Carter, C. Voss, L. Schubert, C. Olah.

  Distill. 2021.
- **The Missing Curve Detectors of InceptionV1: Applying Sparse Autoencoders to InceptionV1 Early Vision** [\[link\]](https://arxiv.org/pdf/2406.03662)

  L. Gorton.

  arXiv preprint arXiv:2406.03662. 2024.

\[19, 20\]

 and place cells in biological brains

- **Place cells, grid cells, and the brain's spatial representation system.** [\[link\]](https://api.semanticscholar.org/CorpusID:16036900)

  E.I. Moser, E. Kropff, M. Moser.

  Annual review of neuroscience, Vol 31, pp. 69-89. 2008.

\[21\]

. In all three of these cases, a continuous variable is represented by a collection of discrete elements that activate for particular value ranges. Moreover, we also observe dilation of the receptive fields (i.e., subsequent features activate over increasingly large character ranges) which is a common characteristic of biological perception of numbers (e.g.,

- **The neural basis of the Weber--Fechner law: a logarithmic mental number line**

  S. Dehaene.

  Trends in cognitive sciences, Vol 7(4), pp. 145--147. Elsevier. 2003.
- **Tuning curves for approximate numerosity in the human intraparietal sulcus**

  M. Piazza, V. Izard, P. Pinel, D. Le Bihan, S. Dehaene.

  Neuron, Vol 44(3), pp. 547--555. Elsevier. 2004.

\[22, 23\]

).

In the [Appendix](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-feature-splitting), we show these features are universal across dictionaries of different sizes, but that some feature splitting occurs with respect to the line width constraint.

#### The Model Represents Character Count on a Continuous Manifold

We observe that character count feature activations rise and fall at an offset, with two features being active at a time for most counts. This pattern suggests that the features are reconstructing a curved continuous manifold, locally parametrized by the activity of the two most active features. Given that their joint activation profiles follow a sinusoidal pattern, we expect reconstructions to lie on a curve between adjacent feature decoders.

To visualize this, we first compute the average layer 2 residual stream for each value of line character count on our synthetic dataset. We compute the PCA of these 150 vectors, and find that the top 6 components capture 95% of the variance; we project data to that 6 dimensional subspace which we call the “character count subspace” (top 3 PCs on the left below, next 3 PCs on the right). We observe the data form a twisting curve, resembling a helix from the perspective of PCs 1–3 and a more complex twist from the perspective of PCs 4–6.

We also reconstruct the residual stream for each datapoint using only the 10 character count features identified above, and compute the average reconstructed residual stream. We project the resulting curve, along with the feature decoders, into the same subspace. We find that the average line character count vectors are quite closely approximated by the feature reconstruction, though with mild kinks near the feature vectors themselves, reminiscent of a spline approximation of a smooth curve. While the 10 feature vectors discretize the curve, interpolating between the 2–3 neighboring features which are active at a time allows for a high-quality reconstruction of 150 data points.

Average Reconstructed DataAverage DataFeature DecodersFeature Reconstruction of Mean ActivationsFirst 3 PCNext 3 PC

Character count is represented on a manifold in a 6 dimensional subspace (jagged line). This manifold can be approximately locally parametrized by the features we identified (crosses).

#### Validation: The Character Count Subspace is Causal

To validate our interpretation of the character count subspace, we perform a coarse-grained ablation and a fine-grained intervention.

Ablation Experiment.  For our ablation experiment, we zero ablate (from a single early layer) a kkk
k-dimensional subspace corresponding to the top kkk
k principal components of the per–character count mean activations and compare to a baseline of ablating a random kkk
k-dimensional subspace. Below we measure the loss effect, broken down by newlines and nonnewlines.

11

Note, in general one should not assume that a subspace spanned by features (or a PCA) is dedicated to those features because it could be in superposition with many other features. However, because in this case the character count subspace is densely active (and therefore less amenable to being in superposition), this experimental design is more justified.

![](<Base64-Image-Removed>)Ablating the character count subspace has a large effect only when the next token is a newline.

Intervention Experiment.  As a more surgical intervention, we perform an experiment to modify the perceived character count at the end of the aluminum prompt (originally 42 characters). Specifically, we sweep over character counts ccc
c, and substitute the mean activation across all tokens in our dataset with count ccc
c. That is, apatched=aoriginal−μoriginal+μca\_{\\text{patched}} = a\_{\\text{original}} - \\mu\_{\\text{original}} + \\mu\_{c}apatched​=aoriginal​−μoriginal​+μc​
a\_{\\text{patched}} = a\_{\\text{original}} - \\mu\_{\\text{original}} + \\mu\_{c} for activation aaa
a and average activation matrix μ\\muμ
\\mu. We perform this intervention for three adjacent early layers and the last two tokens for both the entire mean vector and within the 6 dimensional PCA space of the mean vectors.

12

The attribution graph has several positional features and edges on both the last token (“called”) as well as the second-to-last token (“also”). We change the “also” count representation to be 6 characters prior to that for the final token, to maintain consistency.

![](<Base64-Image-Removed>)Intervening on a rank 6 subspace is sufficient to change the model’s linebreaking behavior.

#### The Probe Perspective

We also train supervised logistic regression probes to predict character count.

13

as a 150-way multiclass classification problem Probes trained after layer 1 achieve a root mean squared error of 5, indicating some intrinsic noise in the character count representation — which is consistent with our features having relatively wide receptive fields. Performing PCA on the 150 probe weight vectors, we find that 6 components capture 82% of the variance.

When we look at the average responses of each probe to tokens with different line character counts, we see a striking pattern. In addition to a diagonal band (probes, like the sparse features, have increasingly wide receptive fields), we see two faint off-diagonal bands on each side! The response curve of each probe is not monotonically decreasing away from its max, but rebounds. This “ringing” turns out to be a natural consequence of embedding a “rippled” manifold into low dimensions.

![](<Base64-Image-Removed>)Response curve of Line Character Count probes as a function of Line Character Count show widening receptive fields and a "ringing" pattern of off-diagonal stripes.

#### [Rippled Representations are Optimal](https://transformer-circuits.pub/2025/linebreaks/index.html\#rippled-representations)

We note that the cosine similarities of the mean activation vector (which form the helix-like curve visualized in PCA space above), the linear probe vectors, and feature decoder vectors all exhibit similar ringing patterns to the above figure.

14

We use the term “ [ringing](https://en.wikipedia.org/wiki/Ringing_(signal))” in the sense of signal processing, a transient oscillation in response to a sharp peak, such as in the Gibbs Phenomenon). Note that not only are neighboring features not orthogonal, features further away have negative similarities, and then those even further away have positive ones again.

![](<Base64-Image-Removed>)

This structure turns out to be a natural consequence of having the desired pattern of similarity, trivially achievable in 150 dimensions, projected down to low dimensions. As a toy model of this, suppose that we wish to have a discretized circle's worth of unit vectors, each similar to its neighbors but orthogonal to those further away. This can be realized by a symmetric set of unit vectors in 150 dimensions with cosine similarity matrix XXX
X pictured below (left). Projecting this to its top 5 eigenvectors yields a 5-dimensional embedding of the same vectors with cosine similarity matrix (below right) exhibiting ringing. We also plot the curve these vectors form in the top 3 eigenvectors. We can think of the original 150-dimensional embedding of the circle as being highly curved, and the resulting 5-dimensional embedding as retaining as much of that curvature as possible. This manifests as ripples in the embedding of the circle when viewed in a 3D projection. A relationship of this construction to Fourier features is discussed in the [appendix](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-gibbs).

![](<Base64-Image-Removed>)Left panel shows an ideal similarity matrix for vectors representing points along a circle. Middle panel shows the optimal (PCA) approximation possible when embedding the points in 5 dimensions. Right panel shows the resulting projection of the circle to the top 3 dimensions, exhibiting rippling.

Alternatively, one can view the ringing from the perspective of sparse feature decoders as a kind of interference weight

- **A Toy Model of Interference Weights** [\[HTML\]](https://transformer-circuits.pub/2025/interference-weights/index.html)

  C. Olah, N.L. Turner, T. Conerly.

   2025\.

\[24\]

. With no capacity constraints, the model might use orthogonal vectors to represent the quantitative response of each feature, with its own receptive field, to the input data. Forced to put them into lower dimensional superposition, the similarity matrix picks up both a wider diagonal stripe and the upper/lower diagonal ringing stripes.

Finally, we also construct a simple physical model showing that the rippling and ringing arise even when the solution is found dynamically, whenever many vectors are packed into a small number of dimensions. Below, we show the result of a simulation in which 100 points confined to a 666
6-dimensional hypersphere are subjected to attractive forces to their 6 closest neighbors on each side (matching the RMSE error of our probes) and repulsive forces to all other points. (To avoid boundary conditions, we use the topology of a circle instead of an interval.) On the right below is a heatmap exhibiting two rings, and on the left is a 3-dimensional projection of the 6-dimensional curve. This simulation is interactive, and the reader is encouraged to experiment with reinitializing the points (↺), switching the ambient dimension, and modifying the width of the attractive zone. Decreasing the attractive zone or increasing the embedding dimension both increase curvature (and the amount of ringing), and vice versa.

15

The simulation can sometimes find itself in local minima. Increasing the width of the attractive zone before decreasing it again usually solves this issue. As the number of points on the curve grows and the attractive zone width shrinks (in relative terms), the curvature grows quite extreme, approaching a space-filling curve in the limit.

N-Sphere Dynamical System

Physical Simulation
Interactive visualization of particle dynamics on an
n-dimensional sphere. Particles attract neighbors and repel
distant points.


▶
⏸


↺

Dimensions:

3D4D5D6D7D8D

Zone Width:
6

Topology:

CircleInterval

Speed: 5

Projection of points

Drag to rotate

Inner Product Matrix

Of particular interest is the result from setting the ambient dimension to 3

16

Optimization in dimension 3, unlike in higher dimensions, admits bad local minima, because a generic curve on the surface of a sphere self-intersects. To avoid this, either increase the zone width until you get a great circle, then decrease it, or do the optimization in 4D, then select 3D.: the result is a curve similar to the seams of a baseball (below left, circle), which matches the topology observed for three intrinsically one-dimensional phenomena observed in

- **The Origins of Representation Manifolds in Large Language Models**

  A. Modell, P. Rubin-Delanchy, N. Whiteley.

  arXiv preprint arXiv:2505.18235. 2025.
- **Not All Language Model Features Are One-Dimensionally Linear** [\[link\]](https://openreview.net/forum?id=d63a4AM4hb)

  J. Engels, E.J. Michaud, I. Liao, W. Gurnee, M. Tegmark.

  The Thirteenth International Conference on Learning Representations. 2025.

\[7, 4\]

, of colors by hue, dates of the year, and years of the 20th century (which also exhibit dilation). Similar ripples were predicted to occur by Olah

- **Feature Manifold Toy Model** [\[link\]](https://transformer-circuits.pub/2023/may-update/index.html#feature-manifolds)

  C. Olah, J. Batson.

   2023\.

\[1\]

 and then observed by Gorton

- **Curve Detector Manifolds in InceptionV1** [\[link\]](https://livgorton.com/curve-detector-manifolds/)

  L. Gorton. 2024.

\[3\]

 in curve detector features in Inception v1. One of the earliest observations of ringing in a cosine similarity plot and rippled spiral/helix shape in a low-dimensional embedding was of the learned positional embeddings of tokens in GPT2

- **{GPT-2}'s positional embedding matrix is a helix** [\[link\]](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix)

  A. Yedidia. 2023.
- **The positional embedding matrix and previous-token heads: how do they actually work?** [\[link\]](https://www.alignmentforum.org/posts/zRA8B2FJLtTYRgie6/the-positional-embedding-matrix-and-previous-token-heads-how)

  A. Yedidia. Alignment Forum. 2023.

\[25, 26\]

. We also find similar structure in other representations, which we study in [More Sensory and Counting Representations](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-sensory) in the appendix.

![](<Base64-Image-Removed>)Left curve is a locally optimal high-curvature embedding of the circle onto the 2-sphere. Right figures, reproduced with permission from Modell et al.

- **The Origins of Representation Manifolds in Large Language Models**

  A. Modell, P. Rubin-Delanchy, N. Whiteley.

  arXiv preprint arXiv:2505.18235. 2025.

\[7\]

, show 3-dimensional PCA projections of data or features related to colours, years, and dates.

### [Sensing the Line Boundary](https://transformer-circuits.pub/2025/linebreaks/index.html\#boundary)

We now study how the character counting representations are used to determine if the current line of text is approaching the line boundary. To detect the line boundary, the model needs to (1) determine the overall line width constraint and (2) compare the current character count with the line width to calculate the characters remaining.

#### Twisting with QK

We find that newline tokens have their own dedicated character [counting features](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-width-features) that activate based on the width of the line, counting the number of characters between adjacent newlines.

To better understand how these representations are related, we train 150 probes for each possible value of “Line Width” like we did for “Character Count”. Using the attribution graph, we identify an attention head which activates boundary detection features. We visualize both sets of counting representations directly using the first 3 components of their joint PCA in the residual stream (left) and in the reduced QK space of this boundaryhead (right).

17

Specifically we multiply the line width probes through WKW\_KWK​
W\_K and the character count probes through WQW\_QWQ​
W\_Q, and plot the points in the 3D PCA basis of their joint embedding.

Character Count ProbesLine Width ProbesCharacter Count through QLine Width through KAlignment Between Character Count Probes and Line Width ProbesIn the residual streamIn boundary head QK space

Boundary heads twist the representation of line width and character count to detect the line boundary.

Left: Joint PCA of character count and line width probes.

Right: Same after multiplying them through the corresponding QK weights of the boundary head. Range is from 40 (dark) to 150 (light).

We find that this attention head “twists” the character count manifold such that character count iii
i is aligned with line width k=i+ϵk=i+\\epsilonk=i+ϵ
k=i+\\epsilon. This causes the head to attend to the newline when the character count is just a bit less than the line width, thereby indicating that the boundary is approaching. This algorithm is quite general, and enables this head to detect approaching line boundaries for arbitrary line widths!

18

This algorithm also generalizes to arbitrary kinds of separators (e.g., double newlines or pipes), as the QK circuit can handle the positional offset independently of the OV circuit copying the separator type.

![](<Base64-Image-Removed>)Cosine similarity of previous line width and character count probes through different transforms. (Left) the identity map, (Center) QK of boundary head, (Right) QK of random head in the same layer. Boundary heads align the probes but with a small offset.

This plot shows that

- In the residual stream, probes for character count iii
  i are maximally aligned with probes line width probes kkk
  k when i=ki=ki=k
  i=k, but are not highly aligned in absolute terms – the maximum cosine sim is ~0.25.
- In the QK space of the boundary head, the probes are maximally aligned on the offdiagonal i<ki < ki<k
  i < k, and are almost perfectly aligned in absolute terms – the maximum cosine sim is ≈1\\approx 1≈1
  \\approx 1.
- In the QK space of a random head, there is almost no structure between the probes.

As a consequence of the ringing in the character count representations, we also observe ringing in the inner products (see [Rippled Representations are Optimal](https://transformer-circuits.pub/2025/linebreaks/index.html#rippled-representations) above). The model is robust to these off-diagonal interference terms via the softmax applied to attention scores.

#### Leveraging Multiple Boundary Heads

We find that the model actually uses multiple boundary heads, each twisting the manifolds by a different offset to implement a kind of “stereoscopic” algorithm for computing the number of characters remaining.

19

There are also multiple sets of boundary heads at multiple layers that usually come in sets of ~3 with similar relative offsets (so not actually “stereo”). We attach more visualizations of boundary heads in the [Appendix](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-twisting).

![](<Base64-Image-Removed>)Cosine similarity of line width and character count probes through three different boundary heads in the same layer with different amounts of twisting. Green line indicates the argmax for each row, and is used to calculate the average offset reported in the subtitles.

To better understand each boundary head’s output, we train a set of probes for each value of characters remaining in the line (i.e., the line width kkk
k minus the character count iii
i, restricted to k−i<40k - i < 40k−i<40
k - i < 40). For each boundary head, we show the proportion of attention on the newline, as well as the norm of each head’s output projected onto the probe space as a function of characters remaining.

As predicted by our weights based analysis, we observe that boundary heads have distinct but overlapping response curves that “tile” the possible values of characters remaining.

![](<Base64-Image-Removed>)Each boundary head’s response curve peaks at a different distance from the end of the line.

It's worth understanding why the model needs multiple boundary heads rather than just one. If the model relied only on boundary head 0, it couldn't distinguish between 5 characters remaining and 17 characters remaining—both would produce similar outputs. By having each head's output vary most significantly in different ranges, their sum achieves high resolution across the entire relevant range of “Characters Remaining” values.

We can see this more clearly by plotting each head's output in the first two principal components of the characters remaining space (which captures 92% of the variance). Head 0 shows large variance in the \[0, 10\] and \[15, 20\] ranges, Head 1 varies most in the \[10, 20\] range, and Head 2 varies most in the \[5, 15\] range. While no single head provides high resolution across the entire curve, their sum produces an evenly spaced representation that covers all values effectively.

![](<Base64-Image-Removed>)Each head’s output as a function of characters remaining, and their sum in the PCA basis. Individual head outputs are almost one-dimensional, while the sum is a two-dimensional curve.

We validate the causal importance of this two-dimensional subspace by performing an ablation and intervention experiment. Specifically, we conduct the same experiments as before: ablate the subspace and measure its effect on loss by token (left) and precisely modulate the characters remaining estimate on the last token in the aluminum prompt by substituting mean activation vectors.

![](<Base64-Image-Removed>)Characters remaining subspace can be causally intervened upon. (Left) Ablating the subspace has a large effect only when the next token is a newline. (Right) We surgically intervene on the characters remaining space to modulate the prediction of the newline by subtracting the true characters remaining mean activation and adding in a patched characters remaining activations. Note that the completion “ aluminum.” requires ten characters to fit.

#### [The Role of the Extra Dimensions](https://transformer-circuits.pub/2025/linebreaks/index.html\#one-dim)

We are now in a position to understand two distinct but related questions: (1) why these counting representations are multidimensional and (2) why multiple attention heads are required to compute these multidimensional representations.

Geometric Computations – A multi-dimensional representation enables the model to rotate position encodings using linear transformations—something impossible with one-dimensional representations. For instance, to detect an approaching line boundary, the model can rotate the position manifold to align with line width, then use a dot product to identify when only a few characters remain. With a 1D encoding, linear operations reduce to scaling and translation, so comparing position against line width would just multiply the two values, producing a monotonically increasing result with no natural threshold. Higher dimensions beyond 2D allow the manifold to pack more information through additional curvature.

Resolution – For character counting, the model must distinguish between adjacent counts for a large range of character positions, as this determines whether the next word fits. In a one-dimensional representation, positions would be arranged along a ray, with each position separated by some constant δ\\deltaδ
\\delta. To reliably distinguish adjacent positions above noise, we need ∣∣v42−v41∣∣=δ\|\|v\_{42} - v\_{41}\|\| = \\delta∣∣v42​−v41​∣∣=δ
\|\|v\_{42} - v\_{41}\|\| = \\delta to exceed some threshold. But with 150+ positions to represent, this creates an untenable choice: either use enormous dynamic range (∣∣v150∣∣≫∣∣v1∣∣\|\|v\_{150}\|\| \\gg \|\|v\_1\|\|∣∣v150​∣∣≫∣∣v1​∣∣
\|\|v\_{150}\|\| \\gg \|\|v\_1\|\|), which is problematic for transformer computations, or sacrifice resolution between adjacent positions. (Normalization blocks only exacerbate this effect: while points can be spaced far away on a ray if their norms get large enough, there is at most π\\piπ
\\pi worth of angular distance along the projection of that ray onto the unit hypersphere.) Embedding the curve into higher dimensions solves this: positions maintain similar norms while being well-separated in the ambient space, achieving fine resolution without norm explosion (See [Rippled Representations are Optimal](https://transformer-circuits.pub/2025/linebreaks/index.html#rippled-representations) above.) For counting the characters remaining, the dynamic range is smaller, and so the model is able to embed the representation in a smaller subspace as a result.

To achieve the curvature for necessary high resolution, multiple attention heads are needed to cooperatively construct the curved geometry of the counting manifold. An individual attention head's output is a linear combination of its inputs (weighted by attention and transformed by the OV circuit), and thus is fundamentally constrained by the curvature already present in those inputs. In the absence of MLP contributions to the counting representation, if the output manifold needs to exhibit substantial curvature, multiple attention heads need to coordinate—each contributing a piece of the overall geometric structure. We will see another example of distributed head computation in the section on the [Distributed Character Counting Algorithm](https://transformer-circuits.pub/2025/linebreaks/index.html#count-algo).

#### [A Discovery Story](https://transformer-circuits.pub/2025/linebreaks/index.html\#discovery)

How did we originally find this boundary detection mechanism? When we first computed an attribution graph, we saw several edges from the previous newline features and embedding to predict-newline features. QK attributions showed that the top key feature was a “the previous line was 40–60 characters long” feature and the top query feature was “the current character count is 35–50” feature. At any one time there were often multiple counting features active at different strengths, suggesting that these features might be discretizing a manifold.

![](<Base64-Image-Removed>)

The boundary heads cause a family of boundary detecting features to activate in response to how close the current line is to the global line width. That is, they sense the approaching line boundary or the reverse index of the line count. Investigating these three sets of feature families led us to the count manifolds which they sparsely parametrize, and investigating the relevant attention heads let us find the boundary heads.

Finally, we note that these boundary-sensing representations parallels a well-studied phenomenon in neuroscience: boundary cells

- **Representation of geometric borders in the entorhinal cortex**

  T. Solstad, C.N. Boccara, E. Kropff, M. Moser, E.I. Moser.

  Science, Vol 322(5909), pp. 1865--1868. American Association for the Advancement of Science. 2008.

\[27\]

, which activate at specific distances from environmental boundaries (e.g., walls). Both the artificial features and biological cells come in families with varied receptive fields and offsets.

### [Predicting the Newline](https://transformer-circuits.pub/2025/linebreaks/index.html\#prediction)

The final step of the linebreak task is to combine the estimate of the line boundary with the prediction of the next word to determine whether the next word will fit on the line, or if the line should be broken.

In the attribution graph for the aluminum prompt, we see exactly this merging of paths. The most influential feature

20

Influence in the sense of influence on the logit node, as defined in Ameisen et al.

- **Circuit Tracing: Revealing Computational Graphs in Language Models** [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

  E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, N.L. Turner, B. Chen, C. Citro, D. Abrahams, S. Carter, B. Hosmer, J. Marcus, M. Sklar, A. Templeton, T. Bricken, C. McDougall, H. Cunningham, T. Henighan, A. Jermyn, A. Jones, A. Persic, Z. Qi, T. Ben Thompson, S. Zimmerman, K. Rivoire, T. Conerly, C. Olah, J. Batson.

  Transformer Circuits. 2025.

\[6\]

 in the entire graph is a late feature that activates in contexts where the next word would cause the current line to exceed the overall line width. For our prompt, this feature upweights the probability of newline and downweights the probability of “aluminum.” The top two inputs to this break predictor feature are a “say aluminum” feature and “boundary detecting” feature that gets activated by the aforementioned boundary head.

![](<Base64-Image-Removed>)

While the boundary detector activates regardless of the next token length, breakpredictorfeatures activate only if the next token will exceed the length of the current line (as in the Aluminum prompt), and hence upweight the prediction of a newline.

21

These features also sometimes activate on zero-width modifier tokens (e.g., a token which indicates the first letter of the following token should be capitalized) that need to be adjacent to the modified token, and the modified token is sufficiently long to go over the line limit (e.g. for “Aluminum” instead of “aluminum”). We also see break suppressor features, which only activate if the next token would just barely fit on the line, and hence downweight the prediction of a newline. Both break predictors and suppressors come in larger feature families, which we display in the [Appendix](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-break-predictors).

![](<Base64-Image-Removed>)Average activations of three features based on true next token character length and the characters remaining in a line (line width − character count).

#### Joint Geometry Enables Easy Computation

What is the geometry underlying the model’s ability to determine if the next token will fit on the line? Put another way, how is the break predictor feature above constructed from the boundary detector and next-word features?

To study this, we compute the average activations at the end of the model (~90% depth) across all tokens for all values of characters remaining iii
i and next token lengths jjj
j.

22

We use the true next non-newline token as the label. This is an approximation because it assumes that the model perfectly predicts the next token. By performing a PCA on the combination of mean vectors, we see that the two counts are arranged in orthogonal subspaces with only moderate curvature. Note, this lower dimensional geometry may suffice here because the dynamic range of the count is much smaller.

Next Token LengthCharacters RemainingNext Token LengthCharacters RemainingMargin After Next WordOrthogonal Representations Create a Linear Decision Boundary for LinebreakingNext Word Length vs Characters RemainingThe Sum Makes Linebreaking Linearly Separable

Low dimensional projections of next token character length and characters remaining counting manifolds for 1 (dark) to 15 (light) characters.

(Left) The PCA of their union. (Right) The PCA of all their pairwise combinations.

The orthogonal representations make the correct newline decision linearly separable.

Now consider the pairwise sum of each possible character-remaining vector iii
i and next-token-length vector jjj
j.

23

This sum is principled because both sets of vectors are marginalized data means, so collectively have the mean of the data, which we center to be 0. Since these counts are arranged orthogonally, the decision to break the line i−j≥0i-j \\geq 0i−j≥0
i-j \\geq 0 corresponds to a simple separating hyperplane. In other words, the prediction to break the line is made trivial by the underlying geometry!

When we use the separating hyperplane from the PCA of these average embeddings on real data, we achieve an AUC of 0.91 on the ground truth of whether the next token should be a newline. This reflects both the error of the three dimensional classifier and the error from Haiku’s estimates of the next token.

If the length of the most likely next word is linearly represented, this scheme would allow the model to predict newlines when that word is longer than the length remaining in the line. One could imagine a more general mechanism where the model comprehensively redirects the probability mass from all words that exceed the line limit to the newline. Claude 3.5 Haiku does not seem to leverage such a mechanism: when we compare the predicted distribution of tokens at the end of a line to the distribution on an identical prompt with the newlines stripped, we find them to be quite different.

### [A Distributed Character Counting Algorithm](https://transformer-circuits.pub/2025/linebreaks/index.html\#count-algo)

Having described how the various character counting representations are used, the last big remaining question is: how are they computed?

We will show how Haiku uses many attention heads across multiple layers to cooperatively compute an increasingly accurate estimate of the character count. This turned out to be the most complicated mechanism we studied, though there are many similarities with the boundary detection mechanism.

To get an intuitive understanding of the behavior of the heads important for counting, we project their outputs into the PCA space of the line character count probes.

24

We display the average outputs over many prompts. Layer 0 heads (left) each write along what appears as a ray when visualized in the first 3 principal components—it is their sum that generates a curved manifold. Layer 1 heads (right) instead output curves which combine to produce an increasingly complex manifold. They appear responsible for sharpening the Layer 0 representation and thus the estimate of the count. We find that the R2R^2R2
R^2 for the character count prediction

25

The prediction is the argmax of the head outputs projected on the character count probes. of the 5 key Layer 0 heads is 0.93, compared to 0.97 using 11 heads in the first two layers.

Head 0Head 1Head 2Head 3Sum of Key HeadsHead 0Head 1Head 2Head 3Sum of Key HeadsIndividual Head Outputs Tile the Joint Output SpaceLayer 0 HeadsLayer 1 Heads

Comparison of Layer 0 (left) vs Layer 1 (right) average attention outputs in the PCA basis of the character count probes from 1 (dark) to 150 (light) characters. In each layer, the outputs from each head tile the space.

In Layer 0, each head output is almost 1-dimensional, while in Layer 1 heads display more curvature (which they got from Layer 0!).

#### Embedding Geometry

To understand how the character count is computed, we start at the very beginning: the embedding matrix.

As before, we can train probes or compute the average weights for every distinct token length in the embedding. We visualize the token character count probes for character length 1–14 and visualize their top principal components. Using the first 3 principal components, which capture 70% of the variance, we see that embedding character counts are arranged in a circular pattern (PC1 vs PC2) with an oscillating component (PC3). This pattern is consistent with the ones observed in [Rippled Representations are Optimal](https://transformer-circuits.pub/2025/linebreaks/index.html#rippled-representations).

![](<Base64-Image-Removed>)PCA of embedding vectors in WEW\_EWE​
W\_E averaged by token character length.

As with all of the counting manifolds, we also find [features](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-token-lengths) that discretize this space into overlapping notions of short, medium, and long words.

#### Attention Head Outputs Sum To Produce the Count

To understand the counting mechanism, we will work backwards from the summed attention outputs to the embedding. Notably, we:

- Ignore MLPs – The attention head outputs affect the character count representation 4× more than the MLPs, so we restrict our focus to attention;
- Focus on First Two Layers – Even after layer 0, counting probes have reasonable accuracy and there are coarse positional features. Therefore, we focus on how attention transforms the embeddings into the count and how layer 1 further refines this representation.

![](<Base64-Image-Removed>)The summed output of 5 important Layer 0 heads on one prompt by token. (Left) The inner product of the summed attention outputs and the character counting probes; (Right) How the argmax of this product compares to the true line count. Context position starts at the first newline, with newlines denoted with dashes.

We can decompose the sum above into the contribution from the output of each individual head in layer 0.

26

We omit a previous token head for visual presentation. Under this lens, we see each head performing a relatively low rank computation akin to a classification.

![](<Base64-Image-Removed>)The individual outputs of 4 important Layer 0 heads on one prompt projected onto the character count probes.

How do individual heads implement this behavior? We can break down the behavior of an individual head by analyzing its QK circuit (where it attends) and OV circuit (the linear transformation from the embeddings to the output)

- **A Mathematical Framework for Transformer Circuits** [\[HTML\]](https://transformer-circuits.pub/2021/framework/index.html)

  N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, N. DasSarma, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, C. Olah.

  Transformer Circuits Thread. 2021.

\[28\]

.

QK Circuit.  Each head hhh
h uses the previous newline as an “attention sink,” such that for some number of tokens after the newline (shs\_hsh​
s\_h), the head just attends to the newline. After shs\_hsh​
s\_h tokens, the head begins to smear its attention over its receptive field, which goes up to a maximum of rhr\_hrh​
r\_h tokens.

![](<Base64-Image-Removed>)The average attention to the previous newline as a function of the token index in the line. Like boundary heads, these counting heads specialize with different positional offsets.

OV Circuit. The OV circuit coordinates with the QK circuit to create a heuristic estimate based on the number of tokens in the line multiplied by the average token length (μc≈4\\mu\_c \\approx 4μc​≈4
\\mu\_c \\approx 4), with an additional length correction term. When attending to the newline, each head upweights the average token length multiplied by the head’s sink size: sh×μcs\_h\\times\\mu\_csh​×μc​
s\_h\\times\\mu\_c characters. If no attention is paid to the newline, then from the perspective of the head, the current token must be at least sh+rhs\_h+r\_hsh​+rh​
s\_h+r\_h tokens into the line and should upweight (sh+rh)×μc(s\_h+r\_h) \\times \\mu\_c(sh​+rh​)×μc​
(s\_h+r\_h) \\times \\mu\_c character outputs. Finally, the OV circuit applies an additional correction depending on whether the tokens in the receptive field are above or below average in length.

Below, we include a detailed walkthrough of L0H1.

![](<Base64-Image-Removed>)The QK and OV circuit of counting head L0H1. Top right: the head output projected onto the character counting probes for 64 tokens of a single prompt (truncated to the first newline). Bottom right: the attention pattern (transposed of the canonical ordering). Top right: the average embedding vectors projected onto the character count probes via the OV matrix. Bottom right: a summary of the overall computation.

For a more detailed analysis of each head, see [The Mechanics of Head Specialization](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-mechanics). Layer 1 attention heads perform a similar operation, but additionally leverage the initial estimate of the character count (see [Layer 1 Head OVs](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-l1-attn)).

#### Computing the Line Width

To compute the line width, the model seems to use a similar distributed counting algorithm to count the characters between adjacent newlines. However, one subtlety that we do not address in this work is how the line width is actually aggregated. It is possible that the model computes a global line width by taking the max over all line lengths in the document or uses an exponentially weighted moving average of the last several line lengths. We do note that the line width uses a partially disjoint set of heads, likely because the “attend to previous newline as a sink” mechanism needs modification when the current token is also a newline.

### [Visual Illusions](https://transformer-circuits.pub/2025/linebreaks/index.html\#illusion)

Humans are susceptible to “visual illusions” in which contextual cues can modulate perception in seemingly unexpected ways. Famous examples include the Müller-Lyer illusion, in which arrows placed on the ends of a line can alter the perceived length of the line

- **The Muller-Lyer illusion explained by the statistics of image--source relationships**

  C.Q. Howe, D. Purves.

  Proceedings of the National Academy of Sciences, Vol 102(4), pp. 1234--1239. National Academy of Sciences. 2005.

\[29\]

; the Ponzo and Sander illusions which also modulate perceived line length

- **A review on various explanations of Ponzo-like illusions**

  G.Y. Yildiz, I. Sperandio, C. Kettle, P.A. Chouinard.

  Psychonomic Bulletin \\& Review, Vol 29(2), pp. 293--320. Springer. 2022.

\[30\]

; and others

- **Space and time in visual context**

  O. Schwartz, A. Hsu, P. Dayan.

  Nature Reviews Neuroscience, Vol 8(7), pp. 522--535. Nature Publishing Group UK London. 2007.

\[31\]

.

![](<Base64-Image-Removed>)Classic visual illusions in which perception of line-length is modulated.

Can we use our understanding of the character counting mechanism to construct a “visual illusion” for language models?

To get started, we took the important attention heads for character counting and investigated what other roles they perform on a wider data distribution. We identified instances in which heads that normally attend from a newline to the previous newline would instead attend from a newline to the two-character string @@. This string occurs as a delimiter in git diffs, a circumstance in which you might want to start your line count at a location other than the newline:

⏎@@-14,30 +31,24 @@ export interface ClaudeCodeIAppTheme {⏎

But what happens when this sequence appears outside of a git diff context—for instance, if we insert @@ in the aluminum prompt without changing the line length?

![](<Base64-Image-Removed>)

We find that it does modulate the predicted next token, disrupting the newline prediction! As predicted, the relevant heads get distracted: whereas with the original prompt, the heads attend from newline to newline, in the altered prompt, the heads also attend to the @@.

![](<Base64-Image-Removed>)Insertion of @@ ‘distracts’ an attention head which normally attends from \\n back to the previous \\n. (left) Original attention pattern (truncated). (right) Attention pattern (truncated) with @@ insertion. Now it also attends back to the @@.

How specific is this result: does any pair of letters nonsensically inserted into the prompt fully disrupt the newline prediction? We analyzed the impact of inserting (at the same two positions) 180 different two-character sequences, half of which were a repeated character. We found that while most inserted sequences moderately impact the probability of predicting a newline, newline usually remains the top prediction. There was also no clear difference between sequences consisting of the same or different characters. However, a few sequences substantially disrupted newline prediction, most of which appeared to be related to code or delimiters of some kind: \`\`>>}};\|\|\|\`,@@.

We further analyzed the extent to which there was a relationship between ‘distraction’ of the important attention heads and the impact on the newline prediction. Indeed we found that many of the sequences with potent modulation of newline probability––and especially code-related character pairs––also exhibited substantial modulation of attention patterns.

![](<Base64-Image-Removed>)Insertion of most pairs of characters only moderately impacts the probability of predicting a newline. A subset of pairs, most of which appear related to code or delimiters, substantially disrupt newline prediction. The impact on newline prediction (originally 0.79) is correlated with how much inserted tokens ‘distracts’ character counting attention heads.

While in the aluminum prompt the task is implicit, this illusion generalizes to settings where the comparison task is made explicit. These direct comparisons are perhaps more analogous to the Ponzo, Sander, and Müller-Lyer illusions, where the perception and comparison is more direct.

![](<Base64-Image-Removed>)

These effects are robust to multiple choice orderings. Moreover, if the length of the text following the @@ exceeds that of the alternative choice, the alternative choice is selected as being shorter.

While we are not claiming any direct analogy between illusions of human visual perception and this alteration of line character count estimates, the parallels are suggestive. In both cases we can see the broader phenomena of contextual cues, and the application of learned priors about those cues, modulating estimates of object properties of entities. In the human case, priors such as three-dimensional perspective can influence perception of object size, or color constancy can influence estimates of luminance (such as in the checker shadow illusion). Here, one possible interpretation of our results is that mis-application of a learned prior, including the role of cues such as @@ in git diffs, can also modulate estimates of properties such as line length.

### [Related Work](https://transformer-circuits.pub/2025/linebreaks/index.html\#related-work)

Objective.This work is at the intersection of LLM “biology” (making empirical observations about what is going on inside models; e.g.

- **On the Biology of a Large Language Model** [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

  J. Lindsey, W. Gurnee, E. Ameisen, B. Chen, A. Pearce, N.L. Turner, C. Citro, D. Abrahams, S. Carter, B. Hosmer, J. Marcus, M. Sklar, A. Templeton, T. Bricken, C. McDougall, H. Cunningham, T. Henighan, A. Jermyn, A. Jones, A. Persic, Z. Qi, T.B. Thompson, S. Zimmerman, K. Rivoire, T. Conerly, C. Olah, J. Batson.

  Transformer Circuits Thread. 2025.
- **A primer in bertology: What we know about how bert works** [\[link\]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00349/96482/A-Primer-in-BERTology-What-We-Know-About-How-BERT)

  A. Rogers, O. Kovaleva, A. Rumshisky.

  Transactions of the Association for Computational Linguistics, Vol 8, pp. 842--866. MIT Press. 2020.

  [DOI: 10.1162/tacl\_a\_00349](https://doi.org/10.1162/tacl_a_00349)

\[32, 33\]

) and low level reverse engineering of neural networks  (attempting to fully characterize an algorithm or mechanism; e.g.

- **Zoom In: An Introduction to Circuits** [\[link\]](https://distill.pub/2020/circuits/zoom-in)

  C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, S. Carter.

  Distill. 2020.

  [DOI: 10.23915/distill.00024.001](https://doi.org/10.23915/distill.00024.001)
- **Interpretability in the wild: a circuit for indirect object identification in gpt-2 small** [\[link\]](https://arxiv.org/pdf/2211.00593)

  K. Wang, A. Variengien, A. Conmy, B. Shlegeris, J. Steinhardt.

  arXiv preprint arXiv:2211.00593. 2022.
- **Progress measures for grokking via mechanistic interpretability** [\[link\]](https://arxiv.org/pdf/2301.05217)

  N. Nanda, L. Chan, T. Lieberum, J. Smith, J. Steinhardt.

  arXiv preprint arXiv:2301.05217. 2023.
- **(How) Do Language Models Track State?**

  B.Z. Li, Z.C. Guo, J. Andreas.

  arXiv preprint arXiv:2503.02854. 2025.

\[34, 35, 36, 37\]

). Methodologically, our work makes heavy use of attribution graphs

- **Circuit Tracing: Revealing Computational Graphs in Language Models** [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

  E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, N.L. Turner, B. Chen, C. Citro, D. Abrahams, S. Carter, B. Hosmer, J. Marcus, M. Sklar, A. Templeton, T. Bricken, C. McDougall, H. Cunningham, T. Henighan, A. Jermyn, A. Jones, A. Persic, Z. Qi, T. Ben Thompson, S. Zimmerman, K. Rivoire, T. Conerly, C. Olah, J. Batson.

  Transformer Circuits. 2025.
- **Automatically identifying local and global circuits with linear computation graphs** [\[link\]](https://arxiv.org/pdf/2405.13868)

  X. Ge, F. Zhu, W. Shu, J. Wang, Z. He, X. Qiu.

  arXiv preprint arXiv:2405.13868. 2024.
- **Transcoders find interpretable LLM feature circuits** [\[PDF\]](http://arxiv.org/pdf/2406.11944.pdf)

  J. Dunefsky, P. Chlenski, N. Nanda.

  Advances in Neural Information Processing Systems, Vol 37, pp. 24375--24410. 2025.

\[6, 38, 39\]

 with QK attributions

- **Tracing Attention Computation Through Feature Interactions** [\[HTML\]](https://transformer-circuits.pub/2025/attention-qk/index.html)

  H. Kamath, E. Ameisen, I. Kauvar, R. Luger, W. Gurnee, A. Pearce, S. Zimmerman, J. Batson, T. Conerly, C. Olah, J. Lindsey.

  Transformer Circuits Thread. 2025.

\[40\]

 built on top of crosscoders

- **Sparse Crosscoders for Cross-Layer Features and Model Diffing** [\[HTML\]](https://transformer-circuits.pub/2024/crosscoders/index.html)

  J. Lindsey, A. Templeton, J. Marcus, T. Conerly, J. Batson, C. Olah.

   2024\.

\[18\]

.

Linebreaking. Michaud et al.

- **The Quantization Model of Neural Scaling** [\[link\]](https://openreview.net/forum?id=3tbTw2ga8K)

  E.J. Michaud, Z. Liu, U. Girit, M. Tegmark.

  Thirty-seventh Conference on Neural Information Processing Systems. 2023.

\[5\]

 identified linebreaking in fixed-width text as one of the top 400 “quanta” of model behavior in the smallest model (70m parameters) in the Pythia suite.

Position.  Prior interpretability work on positional mechanism has largely focused on token position (e.g.,

- **{GPT-2}'s positional embedding matrix is a helix** [\[link\]](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix)

  A. Yedidia. 2023.
- **The positional embedding matrix and previous-token heads: how do they actually work?** [\[link\]](https://www.alignmentforum.org/posts/zRA8B2FJLtTYRgie6/the-positional-embedding-matrix-and-previous-token-heads-how)

  A. Yedidia. Alignment Forum. 2023.
- **Neurons in large language models: Dead, n-gram, positional**

  E. Voita, J. Ferrando, C. Nalmpantis.

  arXiv preprint arXiv:2309.04827. 2023.
- **Understanding positional features in layer 0 {SAE}s** [\[link\]](https://bilalchughtai.co.uk/pos-sae/)

  B. Chughtai, Y. Lau. 2024.
- **Universal neurons in gpt2 language models** [\[link\]](https://arxiv.org/pdf/2401.12181)

  W. Gurnee, T. Horsley, Z.C. Guo, T.R. Kheirkhah, Q. Sun, W. Hathaway, N. Nanda, D. Bertsimas.

  arXiv preprint arXiv:2401.12181. 2024.

\[25, 26, 41, 42, 43\]

). These works have shown that there exist MLP neurons

- **Neurons in large language models: Dead, n-gram, positional**

  E. Voita, J. Ferrando, C. Nalmpantis.

  arXiv preprint arXiv:2309.04827. 2023.
- **Universal neurons in gpt2 language models** [\[link\]](https://arxiv.org/pdf/2401.12181)

  W. Gurnee, T. Horsley, Z.C. Guo, T.R. Kheirkhah, Q. Sun, W. Hathaway, N. Nanda, D. Bertsimas.

  arXiv preprint arXiv:2401.12181. 2024.

\[41, 43\]

, SAE features

- **Understanding positional features in layer 0 {SAE}s** [\[link\]](https://bilalchughtai.co.uk/pos-sae/)

  B. Chughtai, Y. Lau. 2024.

\[42\]

, and learned position embeddings

- **{GPT-2}'s positional embedding matrix is a helix** [\[link\]](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix)

  A. Yedidia. 2023.

\[25\]

 with periodic structure encoding absolute token position. Our work illustrates how a model might also want to construct non-token based position schemes that are more natural for many downstream prediction tasks.

Others have also studied, even going back to LSTMs, the existence of mechanisms in language models for controlling the length of output responses

- **Why neural translations are the right length**

  X. Shi, K. Knight, D. Yuret.

  Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2278--2282. 2016.
- **Length Representations in Large Language Models**

  S. Moon, D. Choi, J. Kwon, H. Kamigaito, M. Okumura.

  arXiv preprint arXiv:2507.20398. 2025.

\[44, 45\]

, as well as performed more theoretical analyses of the space of counting algorithms

- **LSTM networks can perform dynamic counting**

  M. Suzgun, S. Gehrmann, Y. Belinkov, S.M. Shieber.

  arXiv preprint arXiv:1906.03648. 2019.
- **Language models need inductive biases to count inductively**

  Y. Chang, Y. Bisk.

  arXiv preprint arXiv:2405.20131. 2024.

\[46, 47\]

.

Geometry and Feature Manifolds.   Beyond position, there has been extensive work in understanding the geometric representation of numbers, especially in toy models (e.g.,

- **Progress measures for grokking via mechanistic interpretability** [\[link\]](https://arxiv.org/pdf/2301.05217)

  N. Nanda, L. Chan, T. Lieberum, J. Smith, J. Steinhardt.

  arXiv preprint arXiv:2301.05217. 2023.
- **The clock and the pizza: Two stories in mechanistic explanation of neural networks** [\[PDF\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/56cbfbf49937a0873d451343ddc8c57d-Paper-Conference.pdf)

  Z. Zhong, Z. Liu, M. Tegmark, J. Andreas.

  Advances in neural information processing systems, Vol 36, pp. 27223--27250. 2023.
- **Feature emergence via margin maximization: case studies in algebraic tasks**

  D. Morwani, B.L. Edelman, C. Oncescu, R. Zhao, S. Kakade.

  arXiv preprint arXiv:2311.07568. 2023.

\[36, 48, 49\]

) and in the context of arithmetic in LLMs (e.g.,

- **A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis** [\[link\]](https://arxiv.org/pdf/2305.15054)

  A. Stolfo, Y. Belinkov, M. Sachan.

  arXiv preprint arXiv:2305.15054. 2023.
- **Pre-trained large language models use fourier features to compute addition** [\[link\]](https://arxiv.org/pdf/2406.03445)

  T. Zhou, D. Fu, V. Sharan, R. Jia.

  arXiv preprint arXiv:2406.03445. 2024.
- **Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics** [\[link\]](https://arxiv.org/pdf/2410.21272)

  Y. Nikankin, A. Reusch, A. Mueller, Y. Belinkov.

   2024\.
- **Language Models Use Trigonometry to Do Addition** [\[link\]](https://arxiv.org/pdf/2502.00873)

  S. Kantamneni, M. Tegmark. 2025.
- **Understanding In-context Learning of Addition via Activation Subspaces**

  X. Hu, K. Yin, M.I. Jordan, J. Steinhardt, L. Chen.

  arXiv preprint arXiv:2505.05145. 2025.

\[50, 51, 52, 53, 54\]

). Collectively, these works have shown that both real LLMs and toy transformers learn periodic representations

- **Pre-trained large language models use fourier features to compute addition** [\[link\]](https://arxiv.org/pdf/2406.03445)

  T. Zhou, D. Fu, V. Sharan, R. Jia.

  arXiv preprint arXiv:2406.03445. 2024.
- **Language Models Use Trigonometry to Do Addition** [\[link\]](https://arxiv.org/pdf/2502.00873)

  S. Kantamneni, M. Tegmark. 2025.
- **Understanding In-context Learning of Addition via Activation Subspaces**

  X. Hu, K. Yin, M.I. Jordan, J. Steinhardt, L. Chen.

  arXiv preprint arXiv:2505.05145. 2025.

\[51, 53, 54\]

, with numbers arranged in a helix to enable certain matrix multiplication based addition algorithms

- **Progress measures for grokking via mechanistic interpretability** [\[link\]](https://arxiv.org/pdf/2301.05217)

  N. Nanda, L. Chan, T. Lieberum, J. Smith, J. Steinhardt.

  arXiv preprint arXiv:2301.05217. 2023.
- **Language Models Use Trigonometry to Do Addition** [\[link\]](https://arxiv.org/pdf/2502.00873)

  S. Kantamneni, M. Tegmark. 2025.

\[36, 53\]

, and that these representations are provably optimal in certain settings

- **Feature emergence via margin maximization: case studies in algebraic tasks**

  D. Morwani, B.L. Edelman, C. Oncescu, R. Zhao, S. Kakade.

  arXiv preprint arXiv:2311.07568. 2023.

\[49\]

. In our context, we similarly observe helical representations

- **Language Models Use Trigonometry to Do Addition** [\[link\]](https://arxiv.org/pdf/2502.00873)

  S. Kantamneni, M. Tegmark. 2025.

\[53\]

, numeric dilation

- **Number Representations in LLMs: A Computational Parallel to Human Perception**

  H. AlquBoj, H. AlQuabeh, V. Bojkovic, T. Hiraoka, A.O. El-Shangiti, M. Nwadike, K. Inui.

  arXiv preprint arXiv:2502.16147. 2025.

\[55\]

, and distributed algorithms across components that collectively implement a correct computation

- **How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model** [\[PDF\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/efbba7719cc5172d175240f24be11280-Paper-Conference.pdf)

  M. Hanna, O. Liu, A. Variengien.

  Advances in Neural Information Processing Systems, Vol 36, pp. 76033--76060. 2023.
- **Understanding In-context Learning of Addition via Activation Subspaces**

  X. Hu, K. Yin, M.I. Jordan, J. Steinhardt, L. Chen.

  arXiv preprint arXiv:2505.05145. 2025.

\[56, 54\]

.

Multidimensional features with clear geometric structure have been found in more natural contexts

- **Successor Heads: Recurring, Interpretable Attention Heads In The Wild** [\[link\]](https://arxiv.org/pdf/2312.09230)

  R. Gould, E. Ong, G. Ogden, A. Conmy.

   2023\.
- **Not All Language Model Features Are One-Dimensionally Linear** [\[link\]](https://openreview.net/forum?id=d63a4AM4hb)

  J. Engels, E.J. Michaud, I. Liao, W. Gurnee, M. Tegmark.

  The Thirteenth International Conference on Learning Representations. 2025.
- **The Origins of Representation Manifolds in Large Language Models**

  A. Modell, P. Rubin-Delanchy, N. Whiteley.

  arXiv preprint arXiv:2505.18235. 2025.

\[57, 4, 7\]

, like in the representation and computation of certain ordinal relationships (e.g., months of the year). In vision models, curve detector neurons

- **Curve Detectors** [\[link\]](https://distill.pub/2020/circuits/curve-detectors)

  N. Cammarata, G. Goh, S. Carter, L. Schubert, M. Petrov, C. Olah.

  Distill. 2020.

\[58\]

 and features

- **The Missing Curve Detectors of InceptionV1: Applying Sparse Autoencoders to InceptionV1 Early Vision** [\[link\]](https://arxiv.org/pdf/2406.03662)

  L. Gorton.

  arXiv preprint arXiv:2406.03662. 2024.

\[20\]

 have been especially well studied and closely resemble the kind of discretization we observe with the families of character counting features. Many other topics have received interpretability analysis of the underlying geometry, such as grammatical relations

- **A structural probe for finding syntax in word representations** [\[PDF\]](https://aclanthology.org/N19-1419.pdf)

  J. Hewitt, C.D. Manning.

  Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4129--4138. 2019.

  [DOI: 10.18653/v1/N19-1419](https://doi.org/10.18653/v1/N19-1419)
- **Visualizing and measuring the geometry of BERT** [\[PDF\]](https://proceedings.neurips.cc/paper_files/paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf)

  A. Coenen, E. Reif, A. Yuan, B. Kim, A. Pearce, F. Viégas, M. Wattenberg.

  Advances in Neural Information Processing Systems, Vol 32. 2019.

\[10, 11\]

, multilingual representations

- **The geometry of multilingual language model representations**

  T.A. Chang, Z. Tu, B.K. Bergen.

  arXiv preprint arXiv:2205.10964. 2022.

\[12\]

, truth

- **The geometry of truth: Emergent linear structure in large language model representations of true/false datasets** [\[link\]](https://arxiv.org/pdf/2310.06824)

  S. Marks, M. Tegmark.

  arXiv preprint arXiv:2310.06824. 2023.

\[59\]

, binding

- **How do language models bind entities in context?** [\[link\]](https://arxiv.org/pdf/2310.17191)

  J. Feng, J. Steinhardt.

  arXiv preprint arXiv:2310.17191. 2023.

\[60\]

, refusal

- **The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence** [\[link\]](https://arxiv.org/pdf/2502.17420)

  T. Wollschlager, J. Elstner, S. Geisler, V. Cohen-Addad, S. Gunnemann, J. Gasteiger.

  arXiv preprint arXiv:2502.17420. 2025.

\[16\]

, features

- **Projecting assumptions: The duality between sparse autoencoders and concept geometry**

  S.S.R. Hindupur, E.S. Lubana, T. Fel, D. Ba.

  arXiv preprint arXiv:2503.01822. 2025.
- **The geometry of concepts: Sparse autoencoder feature structure**

  Y. Li, E.J. Michaud, D.D. Baek, J. Engels, X. Sun, M. Tegmark.

  Entropy, Vol 27(4), pp. 344. MDPI. 2025.

\[17, 15\]

, and hierarchy

- **The geometry of categorical and hierarchical concepts in large language models**

  K. Park, Y.J. Choe, Y. Jiang, V. Veitch.

  arXiv preprint arXiv:2406.01506. 2024.

\[14\]

, though more conceptual research is needed

- **Relational composition in neural networks: A survey and call to action**

  M. Wattenberg, F.B. Viegas.

  arXiv preprint arXiv:2407.14662. 2024.

\[13\]

.

Perhaps most relevant is recent work from Modell et al.

- **The Origins of Representation Manifolds in Large Language Models**

  A. Modell, P. Rubin-Delanchy, N. Whiteley.

  arXiv preprint arXiv:2505.18235. 2025.

\[7\]

, who provide a more formal notion of a feature manifold, and propose that cosine similarity encodes the intrinsic geometry of features. When testing their theory, they observe highly structured and interpretable data manifolds that have ripples and dilation, similar to our counting manifolds. These observations raise a methodological challenge in how to best capture data with different structure (see e.g.

- **Projecting assumptions: The duality between sparse autoencoders and concept geometry**

  S.S.R. Hindupur, E.S. Lubana, T. Fel, D. Ba.

  arXiv preprint arXiv:2503.01822. 2025.
- **Understanding sparse autoencoder scaling in the presence of feature manifolds** [\[PDF\]](http://arxiv.org/pdf/2509.02565.pdf)

  E.J. Michaud, L. Gorton, T. McGrath.

   2025\.
- **Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning**

  X. Huang, M. Hahn.

  arXiv preprint arXiv:2508.01916. 2025.

\[17, 61, 62\]

), but also the exciting hypothesis that many naturally continuous variables (e.g.,

- **Monotonic representation of numeric properties in language models**

  B. Heinzerling, K. Inui.

  arXiv preprint arXiv:2403.10381. 2024.
- **Language Models Represent Space and Time** [\[link\]](https://arxiv.org/pdf/2310.02207)

  W. Gurnee, M. Tegmark. 2024.

\[63, 64\]

) exist in more organized manifolds.

Biological Analogues.  The geometric and algorithmic patterns we observe have suggestive parallels to perception in biological neural systems. Our character count features are analogous to place cells on a 1-D track

- **Place cells, grid cells, and the brain's spatial representation system.** [\[link\]](https://api.semanticscholar.org/CorpusID:16036900)

  E.I. Moser, E. Kropff, M. Moser.

  Annual review of neuroscience, Vol 31, pp. 69-89. 2008.

\[21\]

 and our boundary detecting features are analogous to boundary cells

- **Representation of geometric borders in the entorhinal cortex**

  T. Solstad, C.N. Boccara, E. Kropff, M. Moser, E.I. Moser.

  Science, Vol 322(5909), pp. 1865--1868. American Association for the Advancement of Science. 2008.

\[27\]

. These features exhibit dilation—representing increasingly large character counts activating over increasingly large ranges—mirroring the dilation of number representations in biological brains

- **The neural basis of the Weber--Fechner law: a logarithmic mental number line**

  S. Dehaene.

  Trends in cognitive sciences, Vol 7(4), pp. 145--147. Elsevier. 2003.
- **Tuning curves for approximate numerosity in the human intraparietal sulcus**

  M. Piazza, V. Izard, P. Pinel, D. Le Bihan, S. Dehaene.

  Neuron, Vol 44(3), pp. 547--555. Elsevier. 2004.

\[22, 23\]

. Moreover, the organization of the features on a low dimensional manifold is an instance of a common motif in biological cognition (e.g.,

- **A neural manifold view of the brain**

  M.G. Perich, D. Narain, J.A. Gallego.

  Nature Neuroscience, pp. 1--16. Nature Publishing Group US New York. 2025.

\[65\]

). While the analogies are not perfect, we suspect that there is still fruitful conceptual overlap from increased collaboration between neuroscience and interpretability

- **Position: An inner interpretability framework for AI inspired by lessons from cognitive neuroscience**

  M.G. Vilas, F. Adolfi, D. Poeppel, G. Roig.

  arXiv preprint arXiv:2406.01352. 2024.
- **Multilevel interpretability of artificial neural networks: leveraging framework and methods from neuroscience**

  Z. He, J. Achterberg, K. Collins, K. Nejad, D. Akarca, Y. Yang, W. Gurnee, I. Sucholutsky, Y. Tang, R. Ianov, others.

  arXiv preprint arXiv:2408.12664. 2024.
- **Cognitively Inspired Interpretability in Large Neural Networks**

  A. Leshinskaya, T. Webb, E. Pavlick, J. Feng, G. Opielka, C. Stevenson, I.A. Blank.

  Proceedings of the Annual Meeting of the Cognitive Science Society, Vol 47. 2025.

\[66, 67, 68\]

.

### [Discussion](https://transformer-circuits.pub/2025/linebreaks/index.html\#discussion)

In this paper, we studied the steps involved in a large model performing a naturalistic behavior. The linebreaking task, frequently encountered in training, requires the model to represent and compute a number of scalar quantities involving position in character count units that are not explicit in its input or output

27

Tokens do not come annotated with character counts, and there are no vertical bars on the page showing the line width., then integrate those values with the outputs of complex semantic circuits (that predict the next proper word) to predict the next token. We found sparse features corresponding to each important step of the computation, and for those steps involving scalar quantities, we were able to find a geometric description that significantly simplified the interpretation of the algorithm used by the model. We now reflect on what we learned from that process:

Naturalistic Behavior and Sensory Processing. Deep mechanistic case studies benefit from choosing behaviors that the model performs consistently well, as these are more likely to have crisper mechanisms. This means prioritizing tasks that are natural in pretraining over tasks that seem natural to human investigators, and ideally, that are easily supervisable. As in biological neuroscience, perceptual tasks are often both natural and easy to supervise for interpretability (e.g., it is easy to modify the input in a programmatic way). Although we sometimes describe the early layers of language models as responsible for “detokenizing” the input

- **Softmax Linear Units** [\[HTML\]](https://transformer-circuits.pub/2022/solu/index.html)

  N. Elhage, T. Hume, C. Olsson, N. Nanda, T. Henighan, S. Johnston, S. ElShowk, N. Joseph, N. DasSarma, B. Mann, D. Hernandez, A. Askell, K. Ndousse, A. Jones, D. Drain, A. Chen, Y. Bai, D. Ganguli, L. Lovitt, Z. Hatfield-Dodds, J. Kernion, T. Conerly, S. Kravec, S. Fort, S. Kadavath, J. Jacobson, E. Tran-Johnson, J. Kaplan, J. Clark, T. Brown, S. McCandlish, D. Amodei, C. Olah.

  Transformer Circuits Thread. 2022.
- **Finding Neurons in a Haystack: Case Studies with Sparse Probing** [\[link\]](https://arxiv.org/pdf/2305.01610)

  W. Gurnee, N. Nanda, M. Pauly, K. Harvey, D. Troitskii, D. Bertsimas.

  arXiv preprint arXiv:2305.01610. 2023.
- **Information flow routes: Automatically interpreting language models at scale**

  J. Ferrando, E. Voita.

  arXiv preprint arXiv:2403.00824. 2024.
- **The remarkable robustness of llms: Stages of inference?**

  V. Lad, J.H. Lee, W. Gurnee, M. Tegmark.

  arXiv preprint arXiv:2406.19384. 2024.

\[69, 70, 71, 72\]

, it is perhaps more evocative to think of this as perception. The beginning of the model is really responsible for seeing the input, and much of the early circuitry is in service of sensing or perceiving the text similar to how early layers in vision models implement low level perception

- **Zoom In: An Introduction to Circuits** [\[link\]](https://distill.pub/2020/circuits/zoom-in)

  C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, S. Carter.

  Distill. 2020.

  [DOI: 10.23915/distill.00024.001](https://doi.org/10.23915/distill.00024.001)
- **Beyond the doors of perception: Vision transformers represent relations between objects**

  M. Lepori, A. Tartaglini, W.K. Vong, T. Serre, B.M. Lake, E. Pavlick.

  Advances in Neural Information Processing Systems, Vol 37, pp. 131503--131544. 2024.

\[34, 73\]

.

The Utility of Geometry.  Many of the representations and computations we studied had elegant geometric interpretations. For example, the counting manifolds are the result of an optimal tradeoff between capacity and resolution, with deep connections to space-filling curves and Fourier features. The boundary head twist was especially beautiful, and after discovering one such head, we were able to correctly predict that there would need to be additional heads to provide curvature in the output. The distributed character counting algorithm was more complex, but we were still able to clarify our view by studying linear actions on these manifolds. For other computations, like the final breaking decision, the linear separation was clearly a part of the story but there must be some additional complexity we were not able to see yet to handle multitoken outputs. For the more semantic operations, we purely relied on the feature view. Of course, describing any behavior in full is immensely complicated, and there is a long list of possible subtleties we did not study: how the model accounts for uncertainty in its counting, its mechanism for estimating the line width given multiple prior lines of text, how it adapts to documents with variable line width, how it handles multiple plausible output tokens of different lengths or multitoken words, or various special cases (e.g., a LaTeX \\footnote{} or a markdown link). For the inspired, we share transcoder attribution graphs for a fixed-width line break prompt on [Gemma 2 2B](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D) and [Qwen 3 4B](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39), using the new neuronpedia interactive interface.

Unsupervised Discovery  It likely would not have been possible to develop this clarity if it were not for the unsupervised sparse features. In fact, when we started this project, we attempted to just probe and patch our way to understanding, but this turned out poorly. Specifically, we did not understand what we were looking for (e.g. we didn’t know to distinguish line width vs. character count), where to look for it (e.g., we didn’t expect line width to only be represented on the newline), or how to look for it (we started by training 1-D linear regression probes). However, after identifying some relevant features but before spending substantial effort systematically characterizing their activity profiles, we were also confused by what they were representing. We saw dozens of features that were vaguely about newlines and linebroken text, but their differences were not obvious from flipping through the activating examples. Only after we tested these features on synthetic datasets did their role in the graph and the underlying computation become clear. We suspect better automatic labels

- **Language models can explain neurons in language models** [\[HTML\]](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)

  S. Bills, N. Cammarata, D. Mossing, H. Tillman, L. Gao, G. Goh, I. Sutskever, J. Leike, J. Wu, W. Saunders.

   2023\.
- **Automatically interpreting millions of features in large language models**

  G. Paulo, A. Mallen, C. Juang, N. Belrose.

  arXiv preprint arXiv:2410.13928. 2024.
- **Enhancing automated interpretability with output-centric feature descriptions**

  Y. Gur-Arieh, R. Mayan, C. Agassy, A. Geiger, M. Geva.

  arXiv preprint arXiv:2501.08319. 2025.

\[74, 75, 76\]

 enhanced with agentic workflows

- **A multimodal automated interpretability agent**

  T.R. Shaham, S. Schwettmann, F. Wang, A. Rajaram, E. Hernandez, J. Andreas, A. Torralba.

  Forty-first International Conference on Machine Learning. 2024.
- **Building and evaluating alignment auditing agents**

  T. Bricken, R. Wang, S. Bowman, E. Ong, J. Treutlein, J. Wu, E. Hubinger, S. Marks.

   2025\.

\[77, 78\]

 would accelerate this work, especially in less verifiable domains.

Feature-Manifold Duality.  ​​The discrete feature and geometric feature-manifold perspectives offer dual lenses on the same underlying object. For example, in this work the model's representation of character count can be completely described (modulo reconstruction error) by the activities of the features we identified, where the action of the boundary heads is described by virtual weights that expand out the feature interactions via attention head matrices. The same character count representation can be described by a 1-dimensional feature manifold – a curve in the residual stream parametrized by the character count variable – where linear action of the boundary heads is described by continuous “twisting” of the manifold. In general, geometric structures learned by the model will likely admit both global parametrizations and local discrete approximations.

The Complexity Tax. Despite this duality, the descriptions produced by the two perspectives differ in their simplicity. The discrete features shatter the model into many pieces, producing a complex understanding of the computation. This seems like a general lesson. It seems like discrete features and attribution graphs may provide a true description of model computation, which can be found in an automated way using dictionary learning. Getting any true, understandable description of the computation is a very non-trivial victory! However, if we stop there, and don't understand additional structure which is present, we pay a complexity tax, where we understand things in a needlessly complicated way. In the line breaking problem, constructing the manifold paid down this tax, but one could imagine other ways of reducing the interpretation burden.

A Call for Methodology. Armed with our feature understanding, we were able to directly search for the relevant geometric structures. This was an existence proof more than a general recipe, and we need methods that can automatically surface simpler structures to pay down the complexity tax. In our setting, this meant studying feature manifolds, and it would be nice to see unsupervised approaches to detecting them. In other cases we will need yet other tools to reduce the interpretation burden, like finding hierarchical representations

- **From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit**

  V. Costa, T. Fel, E.S. Lubana, B. Tolooshams, D. Ba.

  arXiv preprint arXiv:2506.03093. 2025.

\[8\]

 or macroscopic structure

- **Interpretability Dreams** [\[HTML\]](https://transformer-circuits.pub/2023/interpretability-dreams/index.html)

  C. Olah. 2023.

\[9\]

 in the global weights

- **Circuit Tracing: Revealing Computational Graphs in Language Models** [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

  E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, N.L. Turner, B. Chen, C. Citro, D. Abrahams, S. Carter, B. Hosmer, J. Marcus, M. Sklar, A. Templeton, T. Bricken, C. McDougall, H. Cunningham, T. Henighan, A. Jermyn, A. Jones, A. Persic, Z. Qi, T. Ben Thompson, S. Zimmerman, K. Rivoire, T. Conerly, C. Olah, J. Batson.

  Transformer Circuits. 2025.

\[6\]

.

A Call for Biology.  The model must perform other elegant computations. We can find these by starting with a specific task the model performs well, study this from multiple perspectives, develop methodology to answer the remaining questions, and relentlessly attempt to simplify our explanations. Because the investigation is grounded in specific examples of a behavior, it provides a fast feedback loop, can shed light on weaknesses of existing methods and inspire new ones, and can sharpen our conceptual language for understanding neural networks. We would be excited to see more deep case studies that adopt this approach.

### [Citation Information](https://transformer-circuits.pub/2025/linebreaks/index.html\#citation-info)

For attribution in academic contexts, please cite this work as

```
Gurnee, et al., "When Models Manipulate Manifolds: The Geometry of a Counting Task", Transformer Circuits, 2025.
```

BibTeX citation

```
@article{gurnee2025when,
  author={Gurnee, Wes and Ameisen, Emmanuel and Kauvar, Isaac and Tarng ,Julius and Pearce, Adam and Olah, Chris and Batson, Joshua},
  title={When Models Manipulate Manifolds: The Geometry of a Counting Task},
  journal={Transformer Circuits Thread},
  year={2025},
  url={https://transformer-circuits.pub/2025/linebreaks/index.html}
}
```

### [Acknowledgments](https://transformer-circuits.pub/2025/linebreaks/index.html\#acknowledgments)

We would like to thank the following people who reviewed an early version of the manuscript and provided helpful feedback that we used to improve the final version: Owen Lewis, Tom McGrath, Eric Michaud, Alexander Modell, Patrick Rubin-Delanchy, Nicholas Sofroniew, and Martin Wattenberg. We are also thankful to all the members of the interpretability team for their helpful discussion and feedback, especially Doug Finkbeiner for discussions of rippling and ringing, Jack Lindsey on framing, Tom Henighan for feedback on clarity, Brian Chen for improving the design of the figures and line edits of the text, and the team who built the attribution graph

- **Circuit Tracing: Revealing Computational Graphs in Language Models** [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

  E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, N.L. Turner, B. Chen, C. Citro, D. Abrahams, S. Carter, B. Hosmer, J. Marcus, M. Sklar, A. Templeton, T. Bricken, C. McDougall, H. Cunningham, T. Henighan, A. Jermyn, A. Jones, A. Persic, Z. Qi, T. Ben Thompson, S. Zimmerman, K. Rivoire, T. Conerly, C. Olah, J. Batson.

  Transformer Circuits. 2025.

\[6\]

 and QK attribution infrastructure

- **Tracing Attention Computation Through Feature Interactions** [\[HTML\]](https://transformer-circuits.pub/2025/attention-qk/index.html)

  H. Kamath, E. Ameisen, I. Kauvar, R. Luger, W. Gurnee, A. Pearce, S. Zimmerman, J. Batson, T. Conerly, C. Olah, J. Lindsey.

  Transformer Circuits Thread. 2025.

\[40\]

.

### [Haiku Task Performance](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-task-performance)

Haiku is able to adapt to the line length for every value of kkk
k, predicting newlines at the correct positions with high probability by the third line. Of course, some error is to be expected even with a perfect estimate of line length, as the model may incorrectly predict the next semantic token. Below is the mean log-prob and accuracy for newline prediction of Haiku on 200 prose sequences that were synthetically wrapped to have lines of character length kkk
k, for k=20,40,…,140k = 20, 40, \\ldots, 140k=20,40,…,140
k = 20, 40, \\ldots, 140.

![](<Base64-Image-Removed>)

### [Feature Splitting and Universality](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-feature-splitting)

It is natural to ask if the character counting features are fundamental, or simply one discretization of the space among many. We found that dictionaries of different sizes learn features with very similar receptive fields, so this featurization – including the slowly dilating widths – is in some sense canonical. We hypothesize that this canonical structure emerges from boundary constraint: positions near zero (start of line) create a natural anchoring point for feature development.

![](<Base64-Image-Removed>)

The geometry of the decoder directions is also fairly consistent between the dictionaries, showing characteristic ringing.

![](<Base64-Image-Removed>)

However, we do see some evidence of feature splitting. For example, below are three character count feature which activate on the same interval (~20–45 characters in the line), but differentially activate for lines of different widths: LCC2.a activates on all line widths, LCC2.b preferentially activates on long line widths, and LCC2.c preferentially activates when close to the line width boundary.

![](<Base64-Image-Removed>)

Recent work has raised the possibility that feature dictionaries could behave pathologically where there exist feature manifolds

- **Understanding sparse autoencoder scaling in the presence of feature manifolds** [\[PDF\]](http://arxiv.org/pdf/2509.02565.pdf)

  E.J. Michaud, L. Gorton, T. McGrath.

   2025\.

\[61\]

, because a dictionary could allocate an increasing number of features in a finer tiling of the space. However, our observation that cross-coders of varying size tile this feature manifold in a canonical way suggest that this behavior does not occur in this setting.

### [Line Width Features](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-width-features)

Line width features tile the space similar to character count features.

![](<Base64-Image-Removed>)

### [Dynamical System Model](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-dynamical)

We simulate N=100N = 100N=100
N = 100 points on the unit (n−1)(n-1)(n−1)
(n-1)-sphere in Rn\\mathbb{R}^nRn
\\mathbb{R}^n (n∈{3,…,8}n \\in \\{3,\\ldots,8\\}n∈{3,…,8}
n \\in \\{3,\\ldots,8\\}) with pairwise forces: Fij={1−(dij−1)/2rijr^ijwhen dij≤w−min(5,1/rij)rijr^ijwhen dij>w\\mathbf{F}\_{ij} = \\begin{cases} \\frac{1 - (d\_{ij} - 1)/2}{r\_{ij}} \\hat{\\mathbf{r}}\_{ij} & \\text{when }d\_{ij} \\leq w \\\ -\\frac{\\min(5, 1/r\_{ij})}{r\_{ij}} \\hat{\\mathbf{r}}\_{ij} & \\text{when }d\_{ij} > w \\end{cases}Fij​={rij​1−(dij​−1)/2​r^ij​−rij​min(5,1/rij​)​r^ij​​when dij​≤wwhen dij​>w​
\\mathbf{F}\_{ij} = \\begin{cases} \\frac{1 - (d\_{ij} - 1)/2}{r\_{ij}} \\hat{\\mathbf{r}}\_{ij} & \\text{when }d\_{ij} \\leq w \\\ -\\frac{\\min(5, 1/r\_{ij})}{r\_{ij}} \\hat{\\mathbf{r}}\_{ij} & \\text{when }d\_{ij} > w \\end{cases}, where rij=∥xj−xi∥r\_{ij} = \\\|\\mathbf{x}\_j - \\mathbf{x}\_i\\\|rij​=∥xj​−xi​∥
r\_{ij} = \\\|\\mathbf{x}\_j - \\mathbf{x}\_i\\\|, r^ij=(xj−xi)/rij\\hat{\\mathbf{r}}\_{ij} = (\\mathbf{x}\_j - \\mathbf{x}\_i)/r\_{ij}r^ij​=(xj​−xi​)/rij​
\\hat{\\mathbf{r}}\_{ij} = (\\mathbf{x}\_j - \\mathbf{x}\_i)/r\_{ij}, www
w is the attractive zone width parameter, and dij=min(∣j−i∣,∣j−i+N∣,∣j−i−N∣)d\_{ij} = \\min(\|j-i\|, \|j-i+N\|, \|j-i-N\|)dij​=min(∣j−i∣,∣j−i+N∣,∣j−i−N∣)
d\_{ij} = \\min(\|j-i\|, \|j-i+N\|, \|j-i-N\|) is the index distance (for the circular topology; for the interval it is just dij=∣j−i∣d\_{ij} = \|j-i\|dij​=∣j−i∣
d\_{ij} = \|j-i\|). Evolution follows v˙i=∑j≠iFij−0.05vi\\dot{\\mathbf{v}}\_i = \\sum\_{j \\neq i} \\mathbf{F}\_{ij} - 0.05\\mathbf{v}\_iv˙i​=∑j≠i​Fij​−0.05vi​
\\dot{\\mathbf{v}}\_i = \\sum\_{j \\neq i} \\mathbf{F}\_{ij} - 0.05\\mathbf{v}\_i and x˙i=vi\\dot{\\mathbf{x}}\_i = \\mathbf{v}\_ix˙i​=vi​
\\dot{\\mathbf{x}}\_i = \\mathbf{v}\_i with sphere constraint xi←xi/∥xi∥\\mathbf{x}\_i \\leftarrow \\mathbf{x}\_i/\\\|\\mathbf{x}\_i\\\|xi​←xi​/∥xi​∥
\\mathbf{x}\_i \\leftarrow \\mathbf{x}\_i/\\\|\\mathbf{x}\_i\\\| enforced after each timestep (Δt=0.01\\Delta t = 0.01Δt=0.01
\\Delta t = 0.01, damping α=0.95\\alpha = 0.95α=0.95
\\alpha = 0.95).

### [Analytic Construction of Ringing and Fourier Modes](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-gibbs)

We explore a deeper connection between the ringing observed in the character count feature manifold, and a connection to Fourier analysis in an analytical construction.

Suppose that we wish to have a discretized circle's worth of unit vectors, each similar to its neighbors but orthogonal to those further away. Then the cosine similarity matrix XXX
X of these will be the circulant matrix of a narrow-peaked function fff
f (left, below). The columns of the square root of XXX
X are nnn
n vectors v1,…,vn∈Rnv\_1,\\ldots,v\_n \\in \\mathbb{R}^nv1​,…,vn​∈Rn
v\_1,\\ldots,v\_n \\in \\mathbb{R}^n, where nnn
n is the number of discrete points on the circle, whose inner products reproduce the similarity matrix.

28

The entire continuous circle embeds into the infinite-dimensional Hilbert space L2S1L^2\\mathbb{S^1}L2S1
L^2\\mathbb{S^1} via this construction. Now suppose we would like to find vectors in a lower-dimensional space whose similarity matrix approximates XXX
X, like the model does for character counts. The best kkk
k-dimensional approximation of this similarity, in an L2L^2L2
L^2 sense, is given by taking the eigendecomposition of XXX
X and truncating it to the top kkk
k eigenvectors; the square root of the result will provide nnn
n vectors in kkk
k-dimensions with the corresponding similarity pattern. If πk\\pi\_kπk​
\\pi\_k is the projector onto the span of the top kkk
k eigenvectors, then the images πkvi\\pi\_k v\_iπk​vi​
\\pi\_k v\_i, which live in a kkk
k-dimensional subspace, are precisely those vectors. We can see below that the resulting low-rank matrix has ringing ( [colab notebook](https://colab.research.google.com/drive/13L51UzyNQ6SnjNZRyhWsx_QuyQ3LoosW#scrollTo=c1q8ozayknvl)). Finally, because the [discrete Fourier transform diagonalizes circulant matrices](https://en.wikipedia.org/wiki/Circulant_matrix), the Fourier coefficients of fff
f are in fact the principal values of XXX
X; the low-rank approximation consists of truncating the small fourier coefficients of fff
f; and the resulting rows of XXX
X exhibit ringing.

![](<Base64-Image-Removed>)

One essential feature of the representation of line character counts is that the “boundary head” twists the representation, enabling each count to pair with a count slightly larger, indicating that the boundary is close. That is, there is a linear map QK which slides the character count curve along itself. Such an action is not admitted by generic high-curvature embeddings of the circle or the interval like the ones in the physical model we constructed. But it is present in both the manifold we observe in Haiku and, as we now show, in the Fourier construction. First, note permutating the coordinates of Rn\\mathbb{R}^nRn
\\mathbb{R}^n by taking ei↦ei+1e\_i \\mapsto e\_{i+1}ei​↦ei+1​
e\_i \\mapsto e\_{i+1} has the effect of mapping vi↦vi+1v\_i \\mapsto v\_{i+1}vi​↦vi+1​
v\_i \\mapsto v\_{i+1}. That is, the (linear) action of this permutation ρ\\rhoρ
\\rho on Rn\\mathbb{R}^nRn
\\mathbb{R}^n acts by a rotation of the embedded circle with respect to its intrinsic geometry. Because conjugation by ρ\\rhoρ
\\rho fixes the circulant matrix XXX
X, it therefore respects its eigendecomposition, and thus commutes with the projection πk\\pi\_kπk​
\\pi\_k onto the vector space spanned by its top kkk
k eigenvectors. The restriction of ρ\\rhoρ
\\rho to that subspace, ρ‾:=πk∘ρ∘πk\\overline{\\rho}:=\\pi\_k \\circ \\rho \\circ \\pi\_kρ​:=πk​∘ρ∘πk​
\\overline{\\rho}:=\\pi\_k \\circ \\rho \\circ \\pi\_k, acts by rotation on the lower-dimensional vectors ρ‾:πkvi↦πkvi+1\\overline{\\rho}: \\pi\_k v\_i \\mapsto \\pi\_k v\_{i+1}ρ​:πk​vi​↦πk​vi+1​
\\overline{\\rho}: \\pi\_k v\_i \\mapsto \\pi\_k v\_{i+1}. Thus we have found nnn
n vectors in a kkk
k-dimensional space, whose similarity is as close as possible to XXX
X (and has ringing), with a linear action of Rk\\mathbb{R}^kRk
\\mathbb{R}^k that rotates the vectors along a rippled embedded circle.

We evaluate whether a Fourier decomposition of the character count curve is optimal, and find that it is quite close given that it does not account for dilation. Fourier components explain at most 10% less variance than an equivalent number of PCA components, which are optimal for capturing variance.

![](<Base64-Image-Removed>)

Finally, we note that as one moves through layers, the representation becomes more peaked. This sharpening of the receptive field is useful to the model to better estimate character counts, and corresponds to higher curvature in the embedding and, as predicted by the model above, more pronounced ringing. Below we show cross-sections (at character count 30, 60, 90, 120) of the cosine similarity matrix of probes trained after layers 0, 1, 2, and 3. With each subsequent layer, the graphs get more tightly peaked and secondary rings go higher.

![](<Base64-Image-Removed>)

### [Geometry of Twisting](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-twisting)

Different heads access and manipulate the space in different ways. Below, we show the cosine similarity of both probe sets through QK for three heads: one which keeps them aligned, one which shifts character count to align better with later line widths, and one which does the opposite.

![](<Base64-Image-Removed>)

We can also look at this transformation by visualizing the Singular Value Decomposition of each set of probes in a joint basis after passing them through QK. Once more, the alignment, left offset, and right offset can be read directly from the components.

![](<Base64-Image-Removed>)

We can directly plot the first 3 components of the joint probe space after passing them through each QK. Doing so shows that one head keeps the representations aligned, while the others twist them either clockwise or counterclockwise.

![](<Base64-Image-Removed>)

### [Break Predictor Features](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-break-predictors)

Boundary detector features (at about ~⅓ model depth) do not take into account the length of the next token.

![](<Base64-Image-Removed>)

Later in the model, there exist features which incorporate both the number of characters remaining and the length of the most likely next token. These features only activate when the most likely next token is longer than the number of characters remaining (i.e. below the red diagonal below), as is the case in our aluminum prompt.

![](<Base64-Image-Removed>)

We also found features for the converse: features which suppress the newline because the predicted next token is shorter than the number of characters remaining in the line.

![](<Base64-Image-Removed>)

Both break prediction and suppression features sometimes also have interpretable logit effects on the output of all tokens, not just the newline. For instance, the features below respectively excite and suppress the newline as their top effect, but also systematically suppress tokens with more characters. This is because if the model is wrong about the value of the next token (and whether it's a newline), the token must at least be short enough to fit on the line.

![](<Base64-Image-Removed>)

### [Representing Token Lengths](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-token-lengths)

We find layer 0 features that activate as a function of the character count of individual tokens.

![](<Base64-Image-Removed>)

These features are overlapping (e.g. there are tokens for which the long word and medium word features are both active) and non-exhaustive (none of them fire on some common tokens, where we suspect the representation of character length is partially absorbed

- **A is for absorption: Studying feature splitting and absorption in sparse autoencoders** [\[link\]](https://arxiv.org/pdf/2409.14507)

  D. Chanin, J. Wilken-Smith, T. Dulka, H. Bhatnagar, J. Bloom.

  arXiv preprint arXiv:2409.14507. 2024.

\[79\]

 into features which just activate for that token).

### [The Mechanics of Head Specialization](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-mechanics)

Heads collaborate to generate the count manifold, but how does each head aggregate counts?

As a toy model, consider the following construction for character counting with a single attention head:

- The head uses the previous newline token as a “sink” where it defaults all of its attention to (i.e., attention 1)
- Each token since the newline gets α\\alphaα
  \\alpha attention, such that after jjj
  j tokens the newline has 1−αj1 - \\alpha j1−αj
1 - \\alpha j attention. Note this limits the construction to only work on lines of up to 1/α1 / \\alpha1/α
1 / \\alpha tokens, but the model could use multiple heads at different offsets to count for longer sequences. The model could also dedicate attention to tokens proportionally to token length
- The output of the head on the newline is 0, and the output of each non-newline token is a vector with the same direction but magnitude proportional to the character count of the token.

This produces a ray with total length proportional to the character count of the line.

![](<Base64-Image-Removed>)

In practice, we observe that individual attention heads do indeed use the newline as an attention sink, but at different offsets. As an example, we visualize the attention patterns of 4 important Layer 0 heads on several prompts with different line widths (starting from the first newline in the sequence).

![](<Base64-Image-Removed>)Attention patterns of four important Layer 0 heads (columns) for 3 different prompts (rows) prompt showing head specialization. Patterns start from the first newline with red dashes indicating linebreaks.

To characterize the mechanism more precisely, we compute the average attention as a function of the number of tokens since the previous newline and also as a function of the character length of individual tokens.

![](<Base64-Image-Removed>)Normalized attention as a function of tokens since newline (left) and token character length (right) for four heads. Each head specialized in a different offset similar to boundary heads.

Similar to boundary detection, individual attention heads specialize in particular offsets to tile the space. Moreover, we observe that most of these attention heads have a bias towards attending to longer tokens.

In addition to QK, a head can change its output based on the OV circuit

- **A Mathematical Framework for Transformer Circuits** [\[HTML\]](https://transformer-circuits.pub/2021/framework/index.html)

  N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, N. DasSarma, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, C. Olah.

  Transformer Circuits Thread. 2021.

\[28\]

. We study this by analyzing the pairwise interaction of probes as mediated by the OV matrix. Specifically, for the averaged token embedding vectors for each token length EtE\_tEt​
E\_t

29

That is, for each token character length iii
i, we compute the average embedding vector in WEW\_EWE​
W\_E. We also prepend this with the newline embedding vector to make the plot below., our line length probes PcP\_cPc​
P\_c, and the weight matrices for the attention output of each head WOVW\_{OV}WOV​
W\_{OV}, we compute PcTWOVEtP\_c^T W\_{OV} E\_tPcT​WOV​Et​
P\_c^T W\_{OV} E\_t.

![](<Base64-Image-Removed>)Inner product of line character count and token character count through OV for 4 important layer 0 heads. The OV responses reflect the difference in attention pattern biases.

The output of each head can be thought of as having two components: (1) a character offset from the newline driven by the attention pattern and (2) an adjustment based on the actual character length of the tokens. Note that the average character count of a token is approximately 4.5 (and the median is 4), so we can interpret these effects shifting from a mean response (i.e. the transition point is always around count 4).

To walk through a head in action, consider the perspective of L0H1, which attends to the newline for the first ~4 tokens and then spreads out attention over the previous ~4–8 tokens:

- While attending to the newline, L0H1 writes to the 5–20 character count (5–20CC) directions and suppresses the 30–80CC directions. This makes sense as an approximation because attending to the newline implies that the line is currently at most 3 tokens long (newline has no width) and on average, any 3 token span is ~15 characters (and is unlikely to be >30).
- While not attending to the newline at all, L0H1 defaults to predicting CC40, since not attending to the newline implies that there are ~8 tokens in the line with ~5 characters each on average (including spaces). Then, there is an additional correction applied depending on how long the tokens are:

- (1) If tokens being attended to are short (<4 chars), then upweight 10–35CC and downweight >40CC.
- (2) If tokens being attended to are long (≥5 chars) then do the opposite.

- In cases with some attention on newlines and nonnewlines, linearly interpolate the above predictions.

Other heads perform a similar operation, except with different offsets depending on their newline sink behavior. Layer 1 heads also perform a similar operation, though they also can leverage the character count estimate of the Layer 0 heads (see [Layer 1 Head OVs](https://transformer-circuits.pub/2025/linebreaks/index.html#appendix-l1-attn)).

### [Layer 1 Head OVs](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-l1-attn)

Similar to the OVs of the Layer 0 attention heads, Layer 1 heads write to the character count features in accordance to how long the tokens they attend to are.

![](<Base64-Image-Removed>)

However, in addition to the token character length, Layer 1 heads also use the initial line length estimate constructed in Layer 0 to create a more refined estimate of the character count.

![](<Base64-Image-Removed>)

These repeated computations appear responsible for implementing the sharpening of representations.

![](<Base64-Image-Removed>)

### [Full Layer 0 Attention Results](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-full-l0-attn)

Below, we show head sums for 3 different prompts with different line widths.

![](<Base64-Image-Removed>)

As before, we can look at their decomposition.

![](<Base64-Image-Removed>)

### [More Sensory and Counting Representations](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-sensory)

While in this work we carefully studied the perception of line lengths and fixed width text, there are many tasks which language models must perform which benefit from a positional, visual, or spatial representation of the underlying text. In the course of our investigation, we came across several other feature families and representations for these behaviors and report several below.

#### What Follows an Early Linebreak?

In addition to line width for tracking the absolute character length of a full line of text, there also exist features that are sensitive to lines which have ended early (i.e, lines where the character count is substantially shorter than the line width kkk
k). While these features are less useful for linebreaking, they enable the model to better predict the token following a linebreak. Specifically, if a line ends ccc
c characters before the line limit kkk
k, the next word should be at least ccc
c characters, otherwise it would have been able to fit in the previous line.

![](<Base64-Image-Removed>)A feature family for how many characters were remaining in a line after it was broken.

It is worth emphasizing that the role of these features, like others in this work, is not obvious from a typical workflow of quickly looking at dataset examples. It might be tempting to ignore these as "newline" features, but careful analysis yields quite clear behavior.

#### Markdown Table Representations

In addition to prose, language models must parse other kinds of more structured data like tables. Accurate prediction of a table’s content requires careful integration of row and column information (e.g. is this a column of text or numbers?). To facilitate this, we use a synthetic dataset of 20 markdown tables to find feature families which activate on separator tokens, specialized to particular rows or columns. Visualizing feature activations on each of these 20 tables (arranged by location in the table) showed clear patterns.

![](<Base64-Image-Removed>)A feature family for the row index in markdown tables. Activations are shown for 20 tables on the "\|" token in the first column of the nth row.![](<Base64-Image-Removed>)A feature family for the column index in markdown tables. Activations are shown for 20 tables on the "\|" tokens in the nth column.

On a synthetic dataset of larger tables, we also observe counting representations for the column and row index that resemble the character counting representations. Specifically, we see ringing in the pairwise probe cosine similarities and the characteristic “baseball seam” in the PCA basis.

![](<Base64-Image-Removed>)Representations of markdown table row indices (left) and column indices (right). (Top) pairwise inner products of probes trained to predict the index; (bottom) probes projected into a 3D PCA basis.

### [Rejected Titles](https://transformer-circuits.pub/2025/linebreaks/index.html\#appendix-title)

- A General Language Assistant as a Laboratory for A-line-ment
- A-line-ment Science: The Geometry of Textual Perception
- The Geometry of Textual Perception: How Models Stay Aligned
- The Geometry of Counting: How Transformers Perceive and Manipulate Spatial Structure in Text
- The Mechanistic Basis of Alignment
- Reading between the lines: The Perception of Linebreaks
- Linebreaking: More than you wanted to know
- The Line Must Be Drawn Here! Character Counting in Neural Networks
- Newline, Who Dis? Attention Heads and Their Distractible Nature
- The End of the Line: How Language Models Count Characters
- How I Learned to Stop Worrying and Love the Carriage Return
- Breaking down the Linebreak Mechanism: The Geometry of Text Perception
- We Found a GPS Inside a GPT

### Footnotes

01. All features have a magnitude dimension; so a discrete feature is a one-dimensional ray, and a one-dimensional feature manifold is the set of all scalings of that manifold, contracting to the origin. See [What is a Linear Representation? What is a Multidimensional Feature?](https://transformer-circuits.pub/2024/july-update/index.html#linear-representations) [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-1)
02. Michaud et al. looked for “quanta” of model skills by clustering gradients









    - **The Quantization Model of Neural Scaling** [\[link\]](https://openreview.net/forum?id=3tbTw2ga8K)

      E.J. Michaud, Z. Liu, U. Girit, M. Tegmark.

      Thirty-seventh Conference on Neural Information Processing Systems. 2023.

\[5\]

. Their Figure 1 shows that predicting newlines in fixed-width text formed one of the top 400 clusters for the smallest model in the Pythia family, with 70m parameters. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-2)
03. The wrapping constraint is implicit. Each newline gives a lower bound (the previous word did fit) and an upper bound (the next word did not). We do not nail down the extent to which the model performs optimal inference with respect to those constraints, rather focusing on how it approximately uses the length of each preceding line to determine whether to break the next. There are also many edge cases for handling tokenization and punctuation. A model could even attempt to infer whether the source document used a non-monospace font and then use the pixel count rather than the character count as a predictive signal! [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-3)
04. We actually first tried to use patching and probing without looking at the graph as a kind of methodological test of the utility of features, but did not make much progress. In hindsight, we were training probes for quantities different than the ones the model represents cleanly, e.g., a fusion of the current token position and the line width. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-4)
05. Ringing, in the manifold perspective, corresponds to interference in the feature superposition perspective. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-5)
06. Orthogonal dimensions would also not be robust to estimation noise. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-6)
07. Each feature has an encoder, which acts as a linear + (Jump)ReLU probe on the residual stream, and a decoder. Ten features f1,…,f10f\_1,\\ldots,f\_{10}f1​,…,f10​
    f\_1,\\ldots,f\_{10} are associated with line character count. The model's estimate of the character count, given a residual stream vector xxx
    x, is summarized by the set of activities of each of the 10 features {fi(x)}\\{f\_i(x)\\}{fi​(x)}
    \\{f\_i(x)\\}. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-7)
08. The model's estimate of the character count is summarized by the projection π(x)\\pi(x)π(x)
    \\pi(x) of xxx
    x onto that subspace. Two datapoints have similar character counts if their projections are close in that subspace. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-8)
09. The model's estimate of the character count is summarized by the nearest point on the manifold to the projection of xxx
    x into the subspace, and its confidence in that estimate by the magnitude of π(x)\\pi(x)π(x)
    \\pi(x). [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-9)
10. The model's estimate of the character count is summarized by the probability distribution given by the softmax of the probe activities, softmax(Px)(Px)(Px)
    (Px). [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-10)
11. Note, in general one should not assume that a subspace spanned by features (or a PCA) is dedicated to those features because it could be in superposition with many other features. However, because in this case the character count subspace is densely active (and therefore less amenable to being in superposition), this experimental design is more justified. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-11)
12. The attribution graph has several positional features and edges on both the last token (“called”) as well as the second-to-last token (“also”). We change the “also” count representation to be 6 characters prior to that for the final token, to maintain consistency. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-12)
13. as a 150-way multiclass classification problem [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-13)
14. We use the term “ [ringing](https://en.wikipedia.org/wiki/Ringing_(signal))” in the sense of signal processing, a transient oscillation in response to a sharp peak, such as in the Gibbs Phenomenon). [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-14)
15. The simulation can sometimes find itself in local minima. Increasing the width of the attractive zone before decreasing it again usually solves this issue. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-15)
16. Optimization in dimension 3, unlike in higher dimensions, admits bad local minima, because a generic curve on the surface of a sphere self-intersects. To avoid this, either increase the zone width until you get a great circle, then decrease it, or do the optimization in 4D, then select 3D. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-16)
17. Specifically we multiply the line width probes through WKW\_KWK​
    W\_K and the character count probes through WQW\_QWQ​
    W\_Q, and plot the points in the 3D PCA basis of their joint embedding. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-17)
18. This algorithm also generalizes to arbitrary kinds of separators (e.g., double newlines or pipes), as the QK circuit can handle the positional offset independently of the OV circuit copying the separator type. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-18)
19. There are also multiple sets of boundary heads at multiple layers that usually come in sets of ~3 with similar relative offsets (so not actually “stereo”). [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-19)
20. Influence in the sense of influence on the logit node, as defined in Ameisen et al.









    - **Circuit Tracing: Revealing Computational Graphs in Language Models** [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

      E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, N.L. Turner, B. Chen, C. Citro, D. Abrahams, S. Carter, B. Hosmer, J. Marcus, M. Sklar, A. Templeton, T. Bricken, C. McDougall, H. Cunningham, T. Henighan, A. Jermyn, A. Jones, A. Persic, Z. Qi, T. Ben Thompson, S. Zimmerman, K. Rivoire, T. Conerly, C. Olah, J. Batson.

      Transformer Circuits. 2025.

\[6\]

[\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-20)
21. These features also sometimes activate on zero-width modifier tokens (e.g., a token which indicates the first letter of the following token should be capitalized) that need to be adjacent to the modified token, and the modified token is sufficiently long to go over the line limit (e.g. for “Aluminum” instead of “aluminum”). [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-21)
22. We use the true next non-newline token as the label. This is an approximation because it assumes that the model perfectly predicts the next token. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-22)
23. This sum is principled because both sets of vectors are marginalized data means, so collectively have the mean of the data, which we center to be 0. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-23)
24. We display the average outputs over many prompts. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-24)
25. The prediction is the argmax of the head outputs projected on the character count probes. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-25)
26. We omit a previous token head for visual presentation. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-26)
27. Tokens do not come annotated with character counts, and there are no vertical bars on the page showing the line width. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-27)
28. The entire continuous circle embeds into the infinite-dimensional Hilbert space L2S1L^2\\mathbb{S^1}L2S1
    L^2\\mathbb{S^1} via this construction. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-28)
29. That is, for each token character length iii
    i, we compute the average embedding vector in WEW\_EWE​
    W\_E. We also prepend this with the newline embedding vector to make the plot below. [\[↩\]](https://transformer-circuits.pub/2025/linebreaks/index.html#d-footnote-29)

### References

01. Feature Manifold Toy Model [\[link\]](https://transformer-circuits.pub/2023/may-update/index.html#feature-manifolds)

    Olah, C. and Batson, J., 2023.
02. What is a Linear Representation? What is a Multidimensional Feature? [\[link\]](https://transformer-circuits.pub/2024/july-update/index.html#linear-representations)

    Olah, C., 2024.
03. Curve Detector Manifolds in InceptionV1 [\[link\]](https://livgorton.com/curve-detector-manifolds/)

    Gorton, L., 2024.
04. Not All Language Model Features Are One-Dimensionally Linear [\[link\]](https://openreview.net/forum?id=d63a4AM4hb)

    Engels, J., Michaud, E.J., Liao, I., Gurnee, W. and Tegmark, M., 2025. The Thirteenth International Conference on Learning Representations.
05. The Quantization Model of Neural Scaling [\[link\]](https://openreview.net/forum?id=3tbTw2ga8K)

    Michaud, E.J., Liu, Z., Girit, U. and Tegmark, M., 2023. Thirty-seventh Conference on Neural Information Processing Systems.
06. Circuit Tracing: Revealing Computational Graphs in Language Models [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

    Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N.L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Ben Thompson, T., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C. and Batson, J., 2025. Transformer Circuits.
07. The Origins of Representation Manifolds in Large Language Models

    Modell, A., Rubin-Delanchy, P. and Whiteley, N., 2025. arXiv preprint arXiv:2505.18235.
08. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit

    Costa, V., Fel, T., Lubana, E.S., Tolooshams, B. and Ba, D., 2025. arXiv preprint arXiv:2506.03093.
09. Interpretability Dreams [\[HTML\]](https://transformer-circuits.pub/2023/interpretability-dreams/index.html)

    Olah, C., 2023.
10. A structural probe for finding syntax in word representations [\[PDF\]](https://aclanthology.org/N19-1419.pdf)

    Hewitt, J. and Manning, C.D., 2019. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4129--4138. [DOI: 10.18653/v1/N19-1419](https://doi.org/10.18653/v1/N19-1419)
11. Visualizing and measuring the geometry of BERT [\[PDF\]](https://proceedings.neurips.cc/paper_files/paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf)

    Coenen, A., Reif, E., Yuan, A., Kim, B., Pearce, A., Viégas, F. and Wattenberg, M., 2019. Advances in Neural Information Processing Systems, Vol 32.
12. The geometry of multilingual language model representations

    Chang, T.A., Tu, Z. and Bergen, B.K., 2022. arXiv preprint arXiv:2205.10964.
13. Relational composition in neural networks: A survey and call to action

    Wattenberg, M. and Viegas, F.B., 2024. arXiv preprint arXiv:2407.14662.
14. The geometry of categorical and hierarchical concepts in large language models

    Park, K., Choe, Y.J., Jiang, Y. and Veitch, V., 2024. arXiv preprint arXiv:2406.01506.
15. The geometry of concepts: Sparse autoencoder feature structure

    Li, Y., Michaud, E.J., Baek, D.D., Engels, J., Sun, X. and Tegmark, M., 2025. Entropy, Vol 27(4), pp. 344. MDPI.
16. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence [\[link\]](https://arxiv.org/pdf/2502.17420)

    Wollschlager, T., Elstner, J., Geisler, S., Cohen-Addad, V., Gunnemann, S. and Gasteiger, J., 2025. arXiv preprint arXiv:2502.17420.
17. Projecting assumptions: The duality between sparse autoencoders and concept geometry

    Hindupur, S.S.R., Lubana, E.S., Fel, T. and Ba, D., 2025. arXiv preprint arXiv:2503.01822.
18. Sparse Crosscoders for Cross-Layer Features and Model Diffing [\[HTML\]](https://transformer-circuits.pub/2024/crosscoders/index.html)

    Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J. and Olah, C., 2024.
19. Curve Circuits [\[link\]](https://distill.pub/2020/circuits/curve-circuits)

    Cammarata, N., Goh, G., Carter, S., Voss, C., Schubert, L. and Olah, C., 2021. Distill.
20. The Missing Curve Detectors of InceptionV1: Applying Sparse Autoencoders to InceptionV1 Early Vision [\[link\]](https://arxiv.org/pdf/2406.03662)

    Gorton, L., 2024. arXiv preprint arXiv:2406.03662.
21. Place cells, grid cells, and the brain's spatial representation system. [\[link\]](https://api.semanticscholar.org/CorpusID:16036900)

    Moser, E.I., Kropff, E. and Moser, M., 2008. Annual review of neuroscience, Vol 31, pp. 69-89.
22. The neural basis of the Weber--Fechner law: a logarithmic mental number line

    Dehaene, S., 2003. Trends in cognitive sciences, Vol 7(4), pp. 145--147. Elsevier.
23. Tuning curves for approximate numerosity in the human intraparietal sulcus

    Piazza, M., Izard, V., Pinel, P., Le Bihan, D. and Dehaene, S., 2004. Neuron, Vol 44(3), pp. 547--555. Elsevier.
24. A Toy Model of Interference Weights [\[HTML\]](https://transformer-circuits.pub/2025/interference-weights/index.html)

    Olah, C., Turner, N.L. and Conerly, T., 2025.
25. {GPT-2}'s positional embedding matrix is a helix [\[link\]](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix)

    Yedidia, A., 2023.
26. The positional embedding matrix and previous-token heads: how do they actually work? [\[link\]](https://www.alignmentforum.org/posts/zRA8B2FJLtTYRgie6/the-positional-embedding-matrix-and-previous-token-heads-how)

    Yedidia, A., 2023. Alignment Forum.
27. Representation of geometric borders in the entorhinal cortex

    Solstad, T., Boccara, C.N., Kropff, E., Moser, M. and Moser, E.I., 2008. Science, Vol 322(5909), pp. 1865--1868. American Association for the Advancement of Science.
28. A Mathematical Framework for Transformer Circuits [\[HTML\]](https://transformer-circuits.pub/2021/framework/index.html)

    Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S. and Olah, C., 2021. Transformer Circuits Thread.
29. The Muller-Lyer illusion explained by the statistics of image--source relationships

    Howe, C.Q. and Purves, D., 2005. Proceedings of the National Academy of Sciences, Vol 102(4), pp. 1234--1239. National Academy of Sciences.
30. A review on various explanations of Ponzo-like illusions

    Yildiz, G.Y., Sperandio, I., Kettle, C. and Chouinard, P.A., 2022. Psychonomic Bulletin \\& Review, Vol 29(2), pp. 293--320. Springer.
31. Space and time in visual context

    Schwartz, O., Hsu, A. and Dayan, P., 2007. Nature Reviews Neuroscience, Vol 8(7), pp. 522--535. Nature Publishing Group UK London.
32. On the Biology of a Large Language Model [\[HTML\]](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

    Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N.L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T.B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C. and Batson, J., 2025. Transformer Circuits Thread.
33. A primer in bertology: What we know about how bert works [\[link\]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00349/96482/A-Primer-in-BERTology-What-We-Know-About-How-BERT)

    Rogers, A., Kovaleva, O. and Rumshisky, A., 2020. Transactions of the Association for Computational Linguistics, Vol 8, pp. 842--866. MIT Press. [DOI: 10.1162/tacl\_a\_00349](https://doi.org/10.1162/tacl_a_00349)
34. Zoom In: An Introduction to Circuits [\[link\]](https://distill.pub/2020/circuits/zoom-in)

    Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M. and Carter, S., 2020. Distill. [DOI: 10.23915/distill.00024.001](https://doi.org/10.23915/distill.00024.001)
35. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small [\[link\]](https://arxiv.org/pdf/2211.00593)

    Wang, K., Variengien, A., Conmy, A., Shlegeris, B. and Steinhardt, J., 2022. arXiv preprint arXiv:2211.00593.
36. Progress measures for grokking via mechanistic interpretability [\[link\]](https://arxiv.org/pdf/2301.05217)

    Nanda, N., Chan, L., Lieberum, T., Smith, J. and Steinhardt, J., 2023. arXiv preprint arXiv:2301.05217.
37. (How) Do Language Models Track State?

    Li, B.Z., Guo, Z.C. and Andreas, J., 2025. arXiv preprint arXiv:2503.02854.
38. Automatically identifying local and global circuits with linear computation graphs [\[link\]](https://arxiv.org/pdf/2405.13868)

    Ge, X., Zhu, F., Shu, W., Wang, J., He, Z. and Qiu, X., 2024. arXiv preprint arXiv:2405.13868.
39. Transcoders find interpretable LLM feature circuits [\[PDF\]](http://arxiv.org/pdf/2406.11944.pdf)

    Dunefsky, J., Chlenski, P. and Nanda, N., 2025. Advances in Neural Information Processing Systems, Vol 37, pp. 24375--24410.
40. Tracing Attention Computation Through Feature Interactions [\[HTML\]](https://transformer-circuits.pub/2025/attention-qk/index.html)

    Kamath, H., Ameisen, E., Kauvar, I., Luger, R., Gurnee, W., Pearce, A., Zimmerman, S., Batson, J., Conerly, T., Olah, C. and Lindsey, J., 2025. Transformer Circuits Thread.
41. Neurons in large language models: Dead, n-gram, positional

    Voita, E., Ferrando, J. and Nalmpantis, C., 2023. arXiv preprint arXiv:2309.04827.
42. Understanding positional features in layer 0 {SAE}s [\[link\]](https://bilalchughtai.co.uk/pos-sae/)

    Chughtai, B. and Lau, Y., 2024.
43. Universal neurons in gpt2 language models [\[link\]](https://arxiv.org/pdf/2401.12181)

    Gurnee, W., Horsley, T., Guo, Z.C., Kheirkhah, T.R., Sun, Q., Hathaway, W., Nanda, N. and Bertsimas, D., 2024. arXiv preprint arXiv:2401.12181.
44. Why neural translations are the right length

    Shi, X., Knight, K. and Yuret, D., 2016. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2278--2282.
45. Length Representations in Large Language Models

    Moon, S., Choi, D., Kwon, J., Kamigaito, H. and Okumura, M., 2025. arXiv preprint arXiv:2507.20398.
46. LSTM networks can perform dynamic counting

    Suzgun, M., Gehrmann, S., Belinkov, Y. and Shieber, S.M., 2019. arXiv preprint arXiv:1906.03648.
47. Language models need inductive biases to count inductively

    Chang, Y. and Bisk, Y., 2024. arXiv preprint arXiv:2405.20131.
48. The clock and the pizza: Two stories in mechanistic explanation of neural networks [\[PDF\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/56cbfbf49937a0873d451343ddc8c57d-Paper-Conference.pdf)

    Zhong, Z., Liu, Z., Tegmark, M. and Andreas, J., 2023. Advances in neural information processing systems, Vol 36, pp. 27223--27250.
49. Feature emergence via margin maximization: case studies in algebraic tasks

    Morwani, D., Edelman, B.L., Oncescu, C., Zhao, R. and Kakade, S., 2023. arXiv preprint arXiv:2311.07568.
50. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis [\[link\]](https://arxiv.org/pdf/2305.15054)

    Stolfo, A., Belinkov, Y. and Sachan, M., 2023. arXiv preprint arXiv:2305.15054.
51. Pre-trained large language models use fourier features to compute addition [\[link\]](https://arxiv.org/pdf/2406.03445)

    Zhou, T., Fu, D., Sharan, V. and Jia, R., 2024. arXiv preprint arXiv:2406.03445.
52. Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics [\[link\]](https://arxiv.org/pdf/2410.21272)

    Nikankin, Y., Reusch, A., Mueller, A. and Belinkov, Y., 2024.
53. Language Models Use Trigonometry to Do Addition [\[link\]](https://arxiv.org/pdf/2502.00873)

    Kantamneni, S. and Tegmark, M., 2025.
54. Understanding In-context Learning of Addition via Activation Subspaces

    Hu, X., Yin, K., Jordan, M.I., Steinhardt, J. and Chen, L., 2025. arXiv preprint arXiv:2505.05145.
55. Number Representations in LLMs: A Computational Parallel to Human Perception

    AlquBoj, H., AlQuabeh, H., Bojkovic, V., Hiraoka, T., El-Shangiti, A.O., Nwadike, M. and Inui, K., 2025. arXiv preprint arXiv:2502.16147.
56. How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model [\[PDF\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/efbba7719cc5172d175240f24be11280-Paper-Conference.pdf)

    Hanna, M., Liu, O. and Variengien, A., 2023. Advances in Neural Information Processing Systems, Vol 36, pp. 76033--76060.
57. Successor Heads: Recurring, Interpretable Attention Heads In The Wild [\[link\]](https://arxiv.org/pdf/2312.09230)

    Gould, R., Ong, E., Ogden, G. and Conmy, A., 2023.
58. Curve Detectors [\[link\]](https://distill.pub/2020/circuits/curve-detectors)

    Cammarata, N., Goh, G., Carter, S., Schubert, L., Petrov, M. and Olah, C., 2020. Distill.
59. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets [\[link\]](https://arxiv.org/pdf/2310.06824)

    Marks, S. and Tegmark, M., 2023. arXiv preprint arXiv:2310.06824.
60. How do language models bind entities in context? [\[link\]](https://arxiv.org/pdf/2310.17191)

    Feng, J. and Steinhardt, J., 2023. arXiv preprint arXiv:2310.17191.
61. Understanding sparse autoencoder scaling in the presence of feature manifolds [\[PDF\]](http://arxiv.org/pdf/2509.02565.pdf)

    Michaud, E.J., Gorton, L. and McGrath, T., 2025.
62. Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning

    Huang, X. and Hahn, M., 2025. arXiv preprint arXiv:2508.01916.
63. Monotonic representation of numeric properties in language models

    Heinzerling, B. and Inui, K., 2024. arXiv preprint arXiv:2403.10381.
64. Language Models Represent Space and Time [\[link\]](https://arxiv.org/pdf/2310.02207)

    Gurnee, W. and Tegmark, M., 2024.
65. A neural manifold view of the brain

    Perich, M.G., Narain, D. and Gallego, J.A., 2025. Nature Neuroscience, pp. 1--16. Nature Publishing Group US New York.
66. Position: An inner interpretability framework for AI inspired by lessons from cognitive neuroscience

    Vilas, M.G., Adolfi, F., Poeppel, D. and Roig, G., 2024. arXiv preprint arXiv:2406.01352.
67. Multilevel interpretability of artificial neural networks: leveraging framework and methods from neuroscience

    He, Z., Achterberg, J., Collins, K., Nejad, K., Akarca, D., Yang, Y., Gurnee, W., Sucholutsky, I., Tang, Y., Ianov, R. and others,, 2024. arXiv preprint arXiv:2408.12664.
68. Cognitively Inspired Interpretability in Large Neural Networks

    Leshinskaya, A., Webb, T., Pavlick, E., Feng, J., Opielka, G., Stevenson, C. and Blank, I.A., 2025. Proceedings of the Annual Meeting of the Cognitive Science Society, Vol 47.
69. Softmax Linear Units [\[HTML\]](https://transformer-circuits.pub/2022/solu/index.html)

    Elhage, N., Hume, T., Olsson, C., Nanda, N., Henighan, T., Johnston, S., ElShowk, S., Joseph, N., DasSarma, N., Mann, B., Hernandez, D., Askell, A., Ndousse, K., Jones, A., Drain, D., Chen, A., Bai, Y., Ganguli, D., Lovitt, L., Hatfield-Dodds, Z., Kernion, J., Conerly, T., Kravec, S., Fort, S., Kadavath, S., Jacobson, J., Tran-Johnson, E., Kaplan, J., Clark, J., Brown, T., McCandlish, S., Amodei, D. and Olah, C., 2022. Transformer Circuits Thread.
70. Finding Neurons in a Haystack: Case Studies with Sparse Probing [\[link\]](https://arxiv.org/pdf/2305.01610)

    Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D. and Bertsimas, D., 2023. arXiv preprint arXiv:2305.01610.
71. Information flow routes: Automatically interpreting language models at scale

    Ferrando, J. and Voita, E., 2024. arXiv preprint arXiv:2403.00824.
72. The remarkable robustness of llms: Stages of inference?

    Lad, V., Lee, J.H., Gurnee, W. and Tegmark, M., 2024. arXiv preprint arXiv:2406.19384.
73. Beyond the doors of perception: Vision transformers represent relations between objects

    Lepori, M., Tartaglini, A., Vong, W.K., Serre, T., Lake, B.M. and Pavlick, E., 2024. Advances in Neural Information Processing Systems, Vol 37, pp. 131503--131544.
74. Language models can explain neurons in language models [\[HTML\]](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)

    Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J. and Saunders, W., 2023.
75. Automatically interpreting millions of features in large language models

    Paulo, G., Mallen, A., Juang, C. and Belrose, N., 2024. arXiv preprint arXiv:2410.13928.
76. Enhancing automated interpretability with output-centric feature descriptions

    Gur-Arieh, Y., Mayan, R., Agassy, C., Geiger, A. and Geva, M., 2025. arXiv preprint arXiv:2501.08319.
77. A multimodal automated interpretability agent

    Shaham, T.R., Schwettmann, S., Wang, F., Rajaram, A., Hernandez, E., Andreas, J. and Torralba, A., 2024. Forty-first International Conference on Machine Learning.
78. Building and evaluating alignment auditing agents

    Bricken, T., Wang, R., Bowman, S., Ong, E., Treutlein, J., Wu, J., Hubinger, E. and Marks, S., 2025.
79. A is for absorption: Studying feature splitting and absorption in sparse autoencoders [\[link\]](https://arxiv.org/pdf/2409.14507)

    Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H. and Bloom, J., 2024. arXiv preprint arXiv:2409.14507.