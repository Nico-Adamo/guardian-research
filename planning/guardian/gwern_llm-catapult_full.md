# Human-like Neural Nets by Catapulting

Source: <https://gwern.net/llm-catapult>  
Extracted: 2026-06-08  
Original date/status line: 2024-04-21–2026-06-05; finished; certainty: unlikely; importance: 10.

Tags: adversarial examples, grokking (NN), savantism

Speculative proposal to create artificial neural nets with human-like performance by high-learning-rate/regularization training of overparameterized NNs to trigger catapulting/grokking. Over-parameterization as a route to true generalization would resolve many outstanding mysteries of artificial versus natural intelligence.

> There are many mysteries about deep learning and human intelligence, but we could describe the biggest anomaly this way: why are artificial neural nets smart in such stupid ways, and biological brains stupid but in smart ways?
>
> I propose a major change in deep learning scaling paradigms: the architectural differences between human brains and NNs (particularly LLMs) may be due to a bias-variance tradeoff, where LLMs minimize variance and human brains minimize bias. Human brains do this by deep double descent-style overparameterization, and adopting a scaling strategy of extremely high-learning-rate training of extremely overparameterized models on small diverse highly-filtered datasets.
> This approach would lead to sample-efficiently and compute-efficiently traveling (or catapulting) to a highly-generalizing human-like basin in the model loss landscape, while performing poorly up until the end and failing to memorize much data.
>
> If true, this would explain a number of odd stylized facts about how humans/NNs perform well/poorly.
>
> Such a ‘catapulted LLM’ would generalize much better than existing NNs, be immune to adversarial attacks, have better economics and be more resistant to cloning, could potentially enable extremely efficient MLP architectures, and by giving true generalization, provide a sturdy foundation for AI safety in the form of useful NNs which are aligned & safe for the right reasons.
>
> This could be feasibly tested by training multi-trillion-parameter models for relatively few steps at high cyclical learning rate schedules, and benchmarking adversarial and hard examples on tasks like arithmetic and small-image classification.

Because deep learning has continued to scale up and smash through benchmarks and begun to look like it really will be the final AI paradigm, and thus in some sense the same thing as human ‘intelligence’, to a considerable degree, we can regard ‘intelligence’ as solved: intelligence is sufficient compute applied to search over programs (like Turing machines or circuits) to predict or optimize where the optimal solution is a relatively long program.

(This is a companion piece to “Guardian Angels: LLM Personalization for Productivity and Security”.)

## Intelligence, Broadly

A scaling-centric view might be summed up like this:

---

The Master Synthesis

## Anomalies

But this paradigm, as broadly correct as it now seems to be, doesn’t explain everything. We still have many specific problems that this paradigm is too general to explain.

While current NNs, and LLMs in particular, are by far the most human-like AI software ever created, in having human-like strengths and weaknesses, there are a number of anomalies in machine & biological intelligence that have no good answers.

We have many puzzles here, but they all feel connected, somehow.

### Artificial

#### Sample Inefficiency

Why do NNs require Chinchilla-style scaling of data and compute, when humans appear to learn from multiple orders of magnitude less data, and it is increasingly plausible (given various estimates of human-brain equivalents) that they learn from less total compute? Why, as so many connectionist pioneers like Alan Turing expected, do we not train AI like children, with a curriculum and clear developmental stages?

There are many answers offered, none satisfactory. (And what should we make of theoretical results like Rosenfeld 2021’s “Nyquist learners”?)

- **Multi-modality:** while useful, multi-modality has failed to yield any major change of scaling law exponents; unimodal models work shockingly well, and language models turn out to already encode a large amount of visual knowledge and can easily be plugged into vision models (eg. Flamingo, Tsimpoukelli et al 2021).
- **Human sensory input is actually large:** Another common explanation is to deny that humans learn from less data, and argue from raw sensory bandwidth: if vision+sound+touch is such-and-such bits per second and you accumulate over an adult’s lifetime, it can look much more comparable to the trillions of tokens we train an LLM on.

This is unconvincing because the raw sensory bitrate is meaningless: the input is extremely redundant & predictable for the most part. (Imagine sitting in a room staring at a computer screen.) Attempts at quantifying the information content of images, video, or sound, usually indicate that they boil down to the equivalent of a few hundred or thousand tokens and those modalities are easily learned by small models (eg. iGPT / DALL·E 1).

The asymmetry is particularly striking in text-to-image generative models, where the text encoder (usually an afterthought) is often far bigger than the image generator itself.

And on the human side, disabled people are not much less intelligent than normal humans: deaf/blind people are much worse at language tasks, but their fluid intelligence often remains normal. If the sensory bandwidth were so critical, this would be impossible.

- **Active Learning:** human children, unlike models confined to offline imitation learning, can choose what to learn about by exploring their environment or asking questions. In theory, active learning & optimal exploration can be far more sample-efficient than indiscriminate training (exponential rather than power law, at a minimum), and this could account for the entire gap.

However, if we look at the things children actually choose, the data in question doesn’t appear all that amazing. Further, in stark violation of any notion of optimal Bayesian exploration, children often choose to learn on the same data point—eg. watching the same YouTube video hundreds of times.[^1] Or if we watch them ‘explore’ a game or computer, it looks like it is by acting largely at random, and an adult would learn far faster by more carefully thought-out exploration.[^2]

- **Embodiment:** a closely-related topic is the idea of “embodied cognition”, which used to be quite popular as an explanation for the weaknesses of AI—AI models simply lacked commonsense & generalization for lack of a body and an appropriate environment.

But thus far, ‘embodiment’ like training on robotics data (eg. Gato) has exhibited zero transfer to other tasks, never mind massive scaling law gains, and ironically, it is, in fact, embodied tasks like robotics models which have been greatly benefiting from non-embodied pretrained models (including LLMs!).

- **Architecture Magic:** Perhaps in some way, Homo sapiens-style biological neurons are just some near-perfect architecture, and this explains most of the gap; someday we will understand how all artificial neurons are severely hobbled by mistakes that will seem as tragically obvious in hindsight as earlier mistakes like not using backpropagation or using sigmoid activation functions now seem to us, but they remain a mystery for now.

This view was highly plausible until recently, but has been running into many problems.

For starters, we simply have not found any architecture magic.

The most obvious place to find magic would be the learning rule for biological NNs, whatever they use in place of backpropagation… But while people have proposed many biologically-plausible learning rules since Hebb proposed the first learning rule in 1949, which respect the requirements like locality, in every case, those learning rules perform worse than, or at best similar to, backprop! To quote Geoff Hinton:

> So maybe it’s [GPT-4] actually got a much better learning algorithm than us.

And if biological NNs are not so good but there is something special about humans which does make them much better, then why do Homo sapiens not appear to have any major neuroscientific breakthroughs compared to our primate relatives? Why are we so genetically similar, and we have failed in the search for major novel mutations that create humans, and human brains seem increasingly like nothing but “a scaled-up primate brain”? If human (or primate) brains are so uniquely efficiently tuned by evolution, why are bird brains so much more efficient in size & thermodynamics than primate brains, with clear genetic changes, and better scaling to the point where small bird brains like ravens or parrots or vultures exhibit eerie levels of intelligence & behavioral complexity almost on par with dolphins, chimpanzees—or humans? Why does human intelligence exhibit so many bizarre drawbacks or anomalies if it is so optimized, like childhood amnesia or lurching through developmental phases over decades?

If the story of NNs is one of us gradually recapitulating evolution’s perfect neural networks, why does neuroscience provide so little useful inspiration (as is emphasized constantly by neuroscientists & AI researchers, even the “neuroscientific” inspiration for things like self-attention is a very loose inspiration)? Why don’t all major improvements come from success in reverse-engineering the human brain to ever greater biologically-realistic detail? Why is the rapid progress in neuroscience, like scanning entire connectomes, completely irrelevant to cutting-edge NN models? Why do all the scaling laws for CNNs & Transformers look so similar in the exponent, and durable improvement so difficult, to the point where 7 years after Vaswani et al 2017, it is still a relevant baseline? And why, if we have such fundamentally inferior architectures, are the scaling laws so smooth & reliable, instead of breaking frequently or predicted to asymptote at levels far below human?[^3]

- **Conversely, why do NNs provide little insight into biological brains?** In the other direction, neuroscience and individual psychology has hardly benefited from DL; DL has provided tools, and has provided good predictive models of brains, but that is about it.

One could open an issue of a psychometrics or neuroscience journal, and note that, over a decade into the deep learning revolution, if it had never happened, that issue would look about the same and would be completely intelligible to a researcher from 2010.

It is impressive that we can now turn fMRI scans into crude visualizations of what a person is looking at, or we can use LLMs to generate possible questions for surveys, but DL has provided essentially no major insights into such fundamental questions as “what is fluid intelligence? why is the g factor so general? how do neuroanatomic traits like neuron count or network properties cause intelligence or other cognitive abilities?”

Isn’t this astounding? We can now create models of enormous generality like Gato or GPT-4o from scratch to match or exceed humans without the hand-engineering of GOFAI, which seem so eerily human-like in many ways, and which recapitulate so many aspects of human cognition down to heuristics & biases, and our ability to create these artificial intelligences tells us… nothing important about the human brain? Really?

So what, all the DL is just a dead-end and coincidentally capable of all that, and sometime in the future we’ll discover the real route to brain-like intelligence…?

#### Sample Efficiency

Why do NNs require so much data to pretrain, when NNs are as sample-efficient in narrow comparisons?

Despite the huge amount of indiscriminate data used in NN pretraining, we are puzzled because if we examine how well models do in apples-to-apples comparisons, NNs appear to be as roughly as good as humans or biological neural networks at learning from small data.

The simplest answer to the exorbitant data-scaling would be that NNs do a bad job of learning each datapoint—but that doesn’t seem to be true when we compare things like in-context learning (even GPT-3), transfer learning smoothly scaling, child-sized datasets (eg. SAYCam, BabyLM, TinyStories), compare total AlphaZero Go/chess games to total games played by human pros over history, or disable human priors to compare learning speed (cf. human vs Transformer meta-learning), and remarkable experiments like learning an unknown language from a book or raising chickens in virtual reality & comparing their visual capabilities with a Transformer (or vice-versa, have profoundly mistaken beliefs about the world when raised passively in a bubble).

How can we reconcile this apparent contradiction?

#### Smallness

Why are NNs so intelligent, and yet so small?

A NN like GPT-3 is ~0.1t parameters (which are absurdly simplified caricatures of biological neurons), while the human brain is extremely loosely estimated at 100t ‘parameters’, and many neuroscience results are taken as implying that each of those ‘parameters’ translates to thousands of equivalent parameters, at a minimum; an LLM is further disadvantaged by lack of recurrency & online learning, and is generally unable to engage in adaptive computation or ‘ponder’, the way that a brain can continually learn and ruminate.

And yet, despite all these severe disadvantages, LLMs seem too good. It does not feel like a GPT-3 knows or does less than 1/100,000th of what the human brain does; but if we argue that a more plausible number like 1/1,000th is true, that appears to commit us to the position that human brains are highly wasteful of parameter-equivalents (despite the touted superiority of biological brain architecture & evolutionary pressure for efficiency).

And if a GPT-3 really does know that little, then what are those >99,999 things missing from a GPT-3—which each require an entire GPT-3 to handle?

This observation becomes even more extreme as a NN like GPT-3 is well-known to be highly overparameterized and can be shrunk down by orders of magnitude, and it is also known that small NNs can be trained for a long time to achieve performance of large models (albeit compute-inefficiently). We appear to be between Scylla & Charybdis.

Alternate version: why are human brains so overparameterized? Many humans get by with much smaller brains than others; there is a real and causal correlation with brain volume, but even allowing for how crude a proxy that is for the underlying neural nets’ capacity & speed, the effect size seems surprisingly small. Biological brains can survive shocking amounts of physical damage or brain loss, as long as it occurs early in development (eg. hemispherectomy; see also hydrocephalus), and it’s surprisingly hard to find brain lesions which damage IQ instead of specific cognitive abilities. Cross-species, there is clear allometric scaling and the rankings make sense, but the slope is shallow: chimpanzees with less than a third of the brain often seem competitive with some humans in terms of intelligence and ingenuity.

And remarkable instances like Portia spiders or the ability of the dragonfly to home in on prey using barely 16 neurons further raise the question of why so many neurons are necessary (as does, indeed, power law scaling in general).

#### Superhuman Prediction

In particular, why do even tiny LLMs like GPT-2 appear to already be superhuman at next-token prediction in terms of perplexity/bits-per-character, and yet still greatly underperform humans on benchmarks & real-world and benefit in every way from scaling up next-token prediction pretraining?

Next-token prediction pretraining clearly works (and continues to work), and yet, also something is clearly missing from the simple story we tell about it. Humans appear to predict better the more important tokens; why and how?

#### Persistent Adversarial Examples

Why are NN adversarial examples still unsolved over a decade later, while the best efforts at human adversarial examples are debatable?

Scarcely any problems from 2013–2014 DL remain, and yet, in 2024, adversarial examples are not merely still around, but as powerful as ever on standard NN models. Countless solutions have been proposed—and failed. The best defenses are weak & compromise performance, like adversarial training damaging generalization (but oddly, seeming more human-like), or unable to scale, like robustness certification.

SOTA NNs rarely even attempt to reduce adversarial examples, LLMs are highly susceptible (and have rough decision boundaries), and DL practitioners usually accept them as a fact of life & work around them.

Meanwhile, attempts to transfer to humans, or create new ones… work to some degree, but not much, and pushing them to the point of meaningful human error requires such large distortions that they no longer seem meaningful.

### Biological

#### Human Amnesia

Why are humans so forgetful compared to NNs?

If forgetting is critical to flexible decision-making in biological brains like rodent studies, as has been argued, in part by analogy to successful machine learning regularization techniques like weight decay or dropout, why do the most successful NNs still remember so much compared to humans? For example, an LLM like GPT-3 can casually memorize entire pages of text after training on it once.

This is not an extraordinary capability, but happens routinely with LLMs. And the bigger and better they get, the more they are able to memorize.

Whereas with a human, even ones who pride themselves on their memory for text (like myself), memorizing any page of text is effectively impossible without extensive practice in mnemonic techniques, and we tell awed stories of those few human beings who appear to have photographic memories (and demonstrate it is possible for human brains to do such things, they just don’t usually).

The usual human’s lack of a photographic memory, and the shallowness of our understanding, can be demonstrated with hilariously simple examples, like “how is the letter ‘g’ written?” or “which way does Lincoln face on the American penny?” Our memories (and competence in general) are so bad, in fact, that in addition to struggling to remember what we ate for breakfast yesterday or learning basic general knowledge like “the capital of Russia”, we forget almost our entire childhood, and treat this childhood amnesia as completely normal and may even deny that fact.

This is particularly puzzling given that: we never have more neural connections than we do in childhood, as they continually die or are pruned away; it is not universal cross-species; the amnesia doesn’t seem to damage more core knowledge like motor ability or language; important traumatic memories can be retained; and that those lost episodic memories are, to a considerable degree, still there—detectable by implicit measures like lower reaction times or faster recognition of blurred images or needing fewer hints/cues.

Nor is this because memorizing facts is useless to learning; memorization is critical to learning.

But then, if memorizing is so useful and more knowledge is better, why do the most successful human memorizers seem to underperform, and it is a mixed blessing at best? Why are examples of photographic memory anecdotally associated with odd limitations in creativity or generalization, like John von Neumann or Luria’s Solomon Shereshevsky, when experimentally in spaced-repetition memory research, better memorization appears to help generalization & understanding without such drawbacks? Why are normal humans capable of memorizing some things, like the Quran or Paradise Lost or the Homeric epics, given enough time & motivation & tricks like memory palaces, but not routinely?

The most extreme cases are idiot savants, whose overall performance profile and pre-DL descriptions bear an uncanny description to LLMs. The feats of savants are too well-documented to doubt, but are impossible to fit comfortably into standard frameworks: if any humans could defeat LLMs in their strong points, it is savants, but then why does savantism seem to only come with such severe costs?

#### Human Ignorance

And why are humans so ignorant compared to NNs, overall?

Not only do we not recall much easily, we don’t know much either: the breadth of knowledge of even a low-quality LLM in 2024 vastly surpasses that of any individual human. Individual expert humans still usually beat the best LLMs in their narrow niche of expertise, but if we quizzed humans with the same breadth of questions that we benchmark LLMs, I don’t think there is a human alive who would come anywhere near them.

#### Human Intelligence

So why are humans so smart in general, ie. why do humans generalize more robustly (eschewing the use of non-robust features / ‘dimpled’ manifolds like GAN steganography[^4]), are more creative, and largely immune to adversarial examples?

One of the most counterintuitive aspects of NNs is that for all the incredible fluency & genuine capabilities, they nevertheless still occasionally make simple blatant mistakes. LLMs in particular—even excluding technical issues caused by BPEs, RLHF & other safety tuning, greedy sampling, and so on, it remains true that they will make baffling errors which are fatal and cause them to go in circles and fail to generalize for things where there seems to be ample training data.

Another version: given that NNs seem to so easily memorize & overfit (and famously can memorize all training data), and operating by “direct fitting” to ‘interpolate’ between the “underspecified” training data, why do they generalize at all?

#### Need to Sleep

Why do humans need to sleep—so much so that we usually die if sleep-deprived harshly enough?

In contrast, there is no known equivalent of sleep in current DL, nor any apparent need for it.

Even for continual learning or reinforcement learning replay, ordinary minibatch SGD suffices. There is no need for extended periods of offline inactivity or non-interactive computation simply to train on some new data.

#### Stage-Wise Development

Why does human development spurt through stages to a much greater degree than NNs do?

Children seem to often develop in phases, with sudden spurts (Van Der Maas et al 2006; van Geert 1991; cf. phase transition neural dynamics). Further, children say, believe, and do all sorts of crazy or stupid or evil things—while people make fun of the bizarre things LLMs sometimes say, LLMs have nothing on the things little kids routinely say.

(It makes one wonder how human brains reach the adult stage when they apparently must all go through the little kid stages, where if an adult said such things, they would be committed to a mental hospital or feared as a future serial killer.)

That is strikingly different from NNs. NNs demonstrate various kinds of phases within-training and emergence across-scaling, but many of the identified phase transitions, like ‘induction heads’, barely even change the training loss or more than a few benchmarks. In general, NN training appears much smoother than humans do, aside from changes in learning rates, particularly cyclical learning rates.

#### Slow Development

Why does intelligent development take so long?

Since larger models are more sample-efficient, and many intelligent animals (and definitely humans) are massively larger, why does it take so long?

One of the most universal observations in NN scaling is that, whatever the task, “larger parameter-count models are more sample-efficient”: more precisely, they reduce the loss more per data point than a smaller model does, and so, when the training is plotted comparing loss with data processed, larger models show a distinctly ‘steeper’ plunge than small models do.[^5] (This raises important questions that I’m not sure we know the answers to: “How sample-efficient are extremely, extremely large models—multiple orders of magnitude larger than we now train? How large do they have to become before they stop becoming more sample-efficient? If they do stop at some point, can better regularization fix them? Why do we not train these if we are data-limited in some regimes?”)

Since many animal brains appear to be so much vastly more parameterized than any DL model ever, and to have a large compute-budget and be bottlenecked by processing samples sequentially, we would naively expect an even steeper plunge. The expense of intelligence would be running the brain, not wallclock time.

But everywhere we look, like primates or cetaceans or elephants or birds, we see that intelligence tends to be associated with prolonged childhoods & longevity & low reproductive rates.

These long developmental periods don’t seem explicable in purely metabolic terms—you cannot produce a college-level human intelligence by force-feeding toddlers, and while we do see changes like average of puberty dropping like a stone from contemporary caloric abundance, it doesn’t seem to accelerate brain development much.

And why does the development stop? Why don’t we go on being neuroplastic our whole lives but seem to coast after young adulthood, appearing fully functional even if we’ve almost stopped learning like many old people?

##### Human Slowness

Why can human development take so long for many things that other animals learn quickly?

For example, some prey animals are able to walk and even run within minutes or hours of birth (thanks in part to fetal training of motor control via twitching), while a human child is both physically & mentally incapable of crawling for >5,000 hours.

And where are the informative priors from evolution? Evolutionary psychology in particular suggested that many complicated social behaviors could be traced to genes and the human brain would be found to have “massive modularity”; this research program has failed completely.

#### Qualitative Differences in Training Dynamics

Where are the missing equivalents for humans of major NN dynamics like deep double descent (true even with 1,000-degree polynomial splines) & “epoch-wise double descent”, super-convergence, and grokking/catapult or patient-teachers?

And if there are none, why not?

#### Sleep

Tononi’s SHY is one of the most interesting theories attempting to explain sleep in general (rather than dreaming). In SHY, ‘sleep’ is forced by neural net local learning, where the neural net ‘weights’ increase in size over time as they learn; this is bad because it directly increases the energetic demands of firing a synapse, and because (just like an artificial NN), it makes it easier to memorize rather than generalize.

The ‘weights’ need to all be multiplied by a fraction, to shrink them back down to normal energy consumption, and helping regularize the brain; but it can’t be done one by one, because they are all in use.

Sleep, then, is when the entire brain is taken offline, to shrink them all simultaneously. So, you can skip sleep for a while, and your brain keeps chugging along, weights growing slowly and becoming more and more inefficient, but it causes increasing problems, and eventually something diverges.

(Dreams, then are not essential; they are probably more of an enhancement for additional sample-efficiency, by doing a more extreme form of experience replay than can be done during waking.)

#### Savantism

Savants are often capable of remarkable feats of memory or perception, like making near-photorealistic drawings from a single glance at a photograph or memorizing books.

How is it possible for the (non-autistic) Kim Peek to “at age 16–20 months…memorize every book that was read to him. His parents moved Kim’s finger along each sentence being read.”? He could hardly have developed mnemonic palace techniques as an infant or other adult-like memory tricks like hours of repetition; instead, he seems to simply memorize books in a single epoch, so to speak, to parrot them back like an LLM.

Further, why do Peek & other savants seem to avoid childhood amnesia entirely (accounts emphasize that their memory tends to be lifelong, and Peek seems to remember books read as a young child), while also appearing to not go through childhood development stages?

Why do they usually suffer from severe neurological (associated with left hemisphere trauma) & physical problems and demonstrating remarkable deficits in abstract understanding and general intelligence? For example, “he [Peek] cannot, for example, explain many commonplace proverbs”—and even this level of abstraction understanding some proverbs is considered unusual for a savant! As far back as Down 1887, observers were struck by the contrast between the lifelong “verbal adhesion” of many savants and their understanding, or general intelligence.

(Treffert & Wallace 2002 note that of the ~100 well-described cases, the highest IQ was 114.) Strikingly for LLM parallels, savants can also be capable of confabulation: Down 1887 describes an illiterate child who “take up a book…and improvise stories of all kinds with a great deal of skill, and in any variety, to suit the supposed tastes of his auditors”.[^6]

In cases of mathematical calculation prodigies like calendar dates or taking roots, the underlying algorithm can be inferred to some degree, and appear to mix huge amounts of memorization & distilled intuition with explicit algorithms. These all appear to benefit from disabling higher levels of the brain.

Savants are often intellectually-disabled or otherwise abnormal. An important example in this context is Luria’s famous mnemonist, Solomon Shereshevsky (The Mind of a Mnemonist), who simply remembered everything he saw or heard: in part by using extensive visualization & mnemonic approaches like the “memory palace”, but his level of synesthesia also meant that everything was inextricably associated for him.[^7]

He struggled with non-literal interpretations, reading, or even recognizing faces—he could remember exactly how a person’s face looked previously, of course, but then he could not generalize to how their face looked now.

Why did he not benefit from the ability to recall every version of a person’s face, and gain uncanny super-recognizer powers? (See also hyperthymesia: people with HSAM do not have an intrinsically superior memory for learning, but for not forgetting—they seem to struggle to focus on daily life and cope with anxiety or memories of bad days.) He has what sounds like maladaptive daydreaming, and ultimately died of alcoholism.

On a similar note, one of John von Neumann’s famous party tricks was memorizing pages of books, and being able to recite books on command—just one of his many remarkable feats of cognition which have entered legend. But von Neumann was criticized for pedanticism, explicitness, inelegance, brute-force calculation, comfort with following complicated lines of reasoning by reliance on his photographic memory.

Eugene Wigner, who knew Von Neumann so well, while describing his feats, struck a cautionary note: none of them were truly original, and Von Neumann himself reportedly said that “[I] will be forgotten while Kurt Gödel is remembered with Pythagoras”.

Was von Neumann ever original & creative enough to be truly great like Pythagoras or Gödel? Was he held back by his great knowledge & calculating ability (if not exactly “library work”), and unable to drill to the essence of matters like a Grothendieck?

If we combine the ideas of lack of abstraction or later stages of childhood cognitive development, memorization, synesthesia (failing to separate modalities), injuries to the left hemisphere (particularly the temporal lobe), ability to confabulate, and general extreme imbalances in cognitive capabilities compared to normal humans, I am left with the impression that savants are the “LLM version” of humans.

Savants are what can sometimes happen when a key node in the brain is knocked out and higher level processes become less effective, exposing more of the brain’s low-level raw intelligence, which is satisfied by simpler forms of learning like massive memorization.

### Grokking

A good example is the much-discussed phenomenon of “grokking”. In the original grokking paper, a small NN is trained on a simple arithmetic problem; as expected, it quickly memorizes the training data, achieving 0% error while failing to generalize with ~100% error on the held-out data. What is more surprising is that after training for long enough, sometimes, apparently at random it gradually begins to improve on the held-out data despite the ~0% error.

The odds of grokking are improved the more the NN is regularized—such as by oversampling implied data which cannot be easily memorized, and particularly by weight decay, which tries to shrink the size of the NN weights. (The larger the weights, the easier it is to encode information into them, and memorize the training data.) What is going on?

Mechanistically, what seems to happen when it groks: is that the initial hyperparameters are ‘poorly’ chosen and the NN quickly finds a nearby local optimum in the model-space (loss landscape), representing the ‘memorization’ solution, but that after enough training, the weight decay or other regularization makes the local optimum too ‘narrow’ to contain the NN forever, and then it randomly walks out, or is “catapulted” out of the original solution—and eventually reaches a new region of model-space.

This new region still has ~0% training error… but it corresponds to an algorithm completely different from memorization, which is simpler in an intrinsic sense than “memorizing all the training data” and so unsurprisingly, generalizes better, even in the presence of data error.

(It could in fact be many algorithms—there are a surprising number of valid algorithms for that particular arithmetic problem.) And since it is simpler, there are many more possible models which all correspond to the same thing, so this new region of model-space is ‘wider’ than the memorization region, and so the model generally does not randomly walk out of that one; or if it does, it will probably find a third, bigger & better region, and so on.

This is a quite intriguing result, but it’s not obvious how to apply it to anything else, like a full-size LLM. It would doubtless be possible to train an LLM like GPT-3 to zero training error if it was trained for enough epochs—but it would be impossibly expensive, and then you would still have to train long enough past that to trigger “grokking”. Given that it can take hours to trigger “grokking” in even the tiniest possible toy dataset, it is impossible to do true “LLM grokking”.

But the phenomenon of grokking is interesting.

### Cyclical Learning Rates

A related theme is struck by studies of cyclical learning rates, like super-convergence or cosine learning rates, as used in eg. Chinchilla/MiniCPM (which yielded a large improvement over Kaplan et al 2020) and appears critical in continual learning.

Cyclical learning rate schedules look like simulated annealing in having distinct loss curves: regularly spiking up (very) high when the LR goes high, but then dropping rapidly when the LR is lowered and achieving the best results yet. These often correspond to big capability gains; one can think of the high-learning-rate period as ‘exploring’ to create a new model, while the low-learning-rate periods ‘exploit’ and finetune the new model, only to start all over.

(And interestingly, cyclical LRs can be run an indefinite number of times.)

Thus, cyclical LRs can look bad if not properly evaluated, because they trade off short-term loss minimization for long-term gains—if you get them right and run them long enough compared to other LR schedules, “the curves cross”, but until then…

In this respect, cyclical LRs look remarkably like childhood development, complete with periods of high errors while trying to learn new abilities (eg. when children learn sarcasm, it is usually preceded by a period where they make many major errors because they are unable to tell when a negation of an obvious fact is an effective use of sarcasm vs simply false).

### Adversarial Examples & Isoperimetry

Consider also another blessing of scale: adversarial examples and the isoperimetry paper.

Adversarial examples, 10 years after discovery, still are not solved, every defense has been broken, and adversarial examples even transfer in many ways, like across models or modalities. Adversarial examples seem to get weaker with parameter scale, but not necessarily with regular compute-optimal scaling.

The isoperimetry paper advances the thesis that the reason for this is that because current NNs can be seen as subdividing the latent space into linear volumes, defined piecewise, parameter by parameter, the boundaries between each volume are linear and fragile: instead of ‘curving around’ like they ought to, they simply go straight.

Therefore, it is easy to ‘nudge’ any particular data point in a variety of small ways, such that they get pushed just over the artificially simple linear boundaries and into the wrong volume.

The true, correct, curve can be approximated to arbitrary accuracy by simply a lot of small boundary lines, but it may take a lot of them, and thus, a lot of parameters (and data?) to define them all.

This is why all the defenses, like adversarial training, keep failing: they do not solve the basic geometric problem—it’s not a few parameters which are bad, it’s all of them, there are countless boundary lines which are too straight and open for exploitation. Such defenses can, at best, eliminate specific examples and maybe make it harder for a specific attacker to find an example, but they may only push the problem elsewhere, or provide security-by-obscurity.

It is also why examples are so stubborn and transfer: models will tend to conceptualize and think in roughly similar ways, and they are all highly under-parameterized, so they will tend to share similar vulnerabilities.

The apparent message of the isoperimetry paper is that you need to scale up your model sizes. And not by a little: their ballpark estimates suggest that models for things like ImageNet are at least 2 OOMs too small. You can imagine what that must imply about LLMs! A GPT-3-175b-scale dense Transformer model might need to be, say, 100 trillion parameters.

Unfortunately, whether you use Kaplan or Chinchilla scaling laws, a 100t-parameter dense LLM would appear to be out of the question for the foreseeable future. To train a model which is able to fit all of those tiny line-segments defined by those 100t parameters, one by one, would require at least as much data—it would require astronomically more data than plausibly exists, and then the training cost of training a model that size on that much data would be equally astronomical.

At least… if you try to fit those parameters with the standard ‘sharp’ (greedily myopic first-order) SGD, looking at each parameter individually. If instead you crossed your eyes, and stared, all of those pixelated lines might blur into smooth curves, and if you thought about it long enough, suddenly at some point the pattern of curves may snap into the drawing of a Dalmatian dog.

This is how NNs catapulting to new model basins may learn intrinsically more robust models which have the blessings of isoperimetry without the brute-force approach of fitting the smooth curves directly to data.

## How the Brain Works

> [Geoff Hinton] tells of coming home from work one day in a state of great excitement, exclaiming, “I did it! I’ve figured out how the brain works!”
>
> His daughter replied, “Oh Dad, not again!”
>
> —Pedro Domingos (cf. NIPS 2010)

What all of these anomalies seem to share is a core of scaling-law-like relationship of parameters, memorization, generalization, and training—a multi-way bias-variance tradeoff, where different systems hit different points on a Pareto frontier where NNs have low variance in-sample but then high bias out-of-sample or on hard problems, and biological brains are at the other extreme (with an unhappy valley of intermediates which impress no one).

I suggest that the core insight here is that too-extensive memorization is the enemy of abstraction, by leading a model to a local optimum which minimizes error but encodes fundamentally the wrong algorithm. Instead, we must, paradoxically, defy intuitions about overfitting by training as large a model as possible in order to handle as small data as available to a human without overfitting.

What happens when we train a current DL model is that it is lazy, and so it rapidly homes in on the nearest loss local minimum it can find.

This minimum tends to be one where it has highly-efficiently memorized the data and learned all the non-robust features and statistical shortcuts; this assemblage of tricks is genuinely effective and intelligent, and is not “cheating” (non-robust features really are present in the heldout data etc.), but they do not generalize far beyond that, because they have not hit on the true underlying algorithm or latent manifold.

Why do grokking NNs seem to need to “memorize to generalize”? Why might it be hard to make progress towards the true target before the memorization phase?

Perhaps because LLMs both memorize and forget datapoints constantly during training (which is why number of copies in the dataset matters: more likely to have seen it recently before the end)—but this forgetting happens for no good reason? “Repetition is the mother of learning” (which is why spaced repetition is good for abstraction, not just brute facts): it is difficult to generalize even the simplest proposition like A + B = C if a NN is constantly forgetting & relearning each part.

Grokking appears to proceed in two phases, first the memorization of all available training data, then the gradual development internally of an increasingly-refined generalizing algorithm.

With that in mind, we might describe the benefit of the memorization as the “learning facts” phase of pedagogy, and then the generalization phase as “the NN thinking about or pondering the facts it has learned until it gets it”. Each minibatch is another ‘thought’ about the data, as the NN struggles to understand the gestalt of the data as more than a bunch of brute facts, and the gradient descent slowly homes in on the generalizing algorithm.

(And then memorizing too many facts can sabotage grokking by ‘squeezing out’ the generalizing algorithm because too much needs to be memorized—memorizing a few well-chosen examples is more useful than memorizing countless redundant pieces of trivia.)

That true target may be ‘very distant’ in the loss landscape, and getting there may require an exorbitant amount of data—each data point painfully pushing it ever so slightly out of its comfort zone until one day, it finally is forced by the overwhelming weight of long-tail anomalies to turn into the right model.

The right algorithm will lie in a distant part of the model loss-landscape, but to reach it using a reasonable amount of training data requires the model to travel far (as a kind of grokking/catapult/super-convergence), which is only possible if the model is so overparameterized that it can encode smooth paths (like saddle-points, as all models are linear mode connected seemingly) and it can ensemble over extremely large families of models which ‘blur out’ to a smooth abstraction of the posterior.

And then high-learning-rate training is critical to kick it along the frequent saddle-points & plains that slow down optimization, oscillating between high learning rates to escape the current local optimum and lower learning rates to consolidate the gains and find the new escape route.

These models inherently require long serial training with many time-steps, and cannot be easily ‘parallelized’ or absorb large amounts of information, and benefit from many long periods of inactivity (‘sleep’) to globally repair the damage from learning or ‘catch up’ on backlogged steps.

Such a model will memorize little of its training data, because that would require rigid, fragile, precise parameters—but those parameters all need to be recycled and explore strange new model loss-landscapes in order to eventually arrive at the promised land.

So during training, and even afterwards, such a model will forget and perform badly on benchmarks that reward memorization (such as declarative knowledge)—even though it will avoid adversarial examples (because none of its boundaries are extremely low-parameter linear lines dependent on inhumanly high-frequency texture-biased non-robust features) and will generalize well to hard problems (which by definition make up little of the standard benchmarks) by learning all sub-skills and achieving ‘slingshot generalization’ by mastering all combinations of skills, and resolve many issues with contemporary DL.

Such models will be hard to discover because of the use of early stopping and the general greediness of DL training (and R&D in general), even though the core ideas are well-known and have large associated literatures: like the original DL scaling ideas, the payoff of catapulting simply takes too long to come.

This intelligence is not that hard to evolve[^8], but it usually is not worthwhile. So they fall into an uneasy ecological niche: such robust generalization is not useful for many animals compared to simpler, cheaper, forms of learning, such as association or imitation, or hardwiring directly into genetics.

It can only pay back its exorbitant costs if the environment rewards robust generalization appropriately by providing enough high-payoff opportunities which are predictable but not too predictable.

## Training a Catapulted LLM
What Is the Largest Useful LLM?

LLMs are more sample-efficient the larger they are. At what parameter scale does this stop holding true? And why don’t we know the answer to this question already?

What would a ‘human-sized LLM’ (or HLLM) look like?
First, it would need to be highly overparameterized relative to the problem as a whole, in order to provide a smoothly-connected loss landscape.
Second, because it is so overparameterized, standard hyperparameters will result in overfitting but not catapulting; heavy regularization will be required, and weight decay is probably not enough (as it doesn’t result in much model movement) but a high learning rate schedule may be adequate regularization given prior art like super-convergence & human parallels.
The final full-scale HLLM would probably look something like a dense Transformer or MLP which is multiple orders of magnitude larger than currently trained[^9] (so >10-trillion parameters, possibly >100-trillion[^10]), in order to be highly-overparameterized compared to the full distribution.

(In grokking papers or the isoperimetry estimates, the NNs are generally several OOMs larger than ‘reasonable’, so if we ballpark GPT-4-level models at ~1t, we would weakly expect the catapulting regime to be ~100t.) The NN should probably be a “skinny” one, emphasizing depth over width; a long-standing trend in DL is that ‘wide’ networks tend to memorize more heavily and poorly express more algorithmic/computational reasoning, and that DL NNs tend to look rather un-biological in using so little recurrence/iterative computation or composing reasoning (while RNNs often do better than Transformers in specific cases, generalizing better despite their overall inferiority).[^11] Very deep networks tend to be avoided due to overfitting or instability rendering their theoretical advantages moot, but catapulting would potentially fix that, and benefit from the inductive biases.

This is consistent with the most recent work on sample-efficient LLMs, like Kim et al 2025 or NanoGPT Slowrun, which emphasize increasing parameter size (eg. via ensembling) and heavy regularization over many epochs.

The NN is trained on small text corpuses like BabyLM scale (~0.1b words). One benefit here is that because the sample size has to be limited, that means it can be filtered extremely heavily for quality/deduplication: because the data distribution decides whether a model will meta-learn, the dataset should be as diverse as possible, and penalize memorization as much as possible (see Wang et al 2024).

Each minibatch should sample as many distinct datapoints as possible, and likewise, diversity maximized across batches, so each catapulting step is catapulting for as many ‘skills’ or ‘capabilities’ simultaneously as possible. This will delay their learning, because only the bare minimum of data is available per skill, so there is no overkill, and they will tend to be learned simultaneously—which leads to ‘emergence’ as multi-step processes suddenly start becoming possible.

(Because humans can replay memories, both in short-term and long-term memory and through the hippocampus, it would be reasonable to do multiple epochs if there is not enough high-quality diverse data for single-epoch training.)

The catapulting itself is due to a cyclical learning rate schedule like super-convergence, perhaps combined with heavy weight decay.[^12]

So, what would happen when training our oversized LLM on our highly-diverse memorization/repetition-purified data with cyclical schedules for heavy regularization? We would observe the classic cyclical training loss behaviors of spikes followed by reaching new lows, but with stagnant performance on the truly held-out data, as the LLM goes through the memorization phase, and eventually reaching a new regime where it begins to transition from memorization to generalization over many tasks simultaneously, which will then suddenly ‘emerge’.

Each cycle builds up a new set of atomic skills, dependent on the skills learned in previous cycles (analogous to developmental phases).

### Prototyping With Arithmetic

It might be feasible to test LLM catapulting on small-scale tasks where current LLMs clearly generalize poorly, like arithmetic. Arithmetic is about the smallest, simplest, easiest-to-generate problem that LLMs still fail in oddly brittle ways on, so it’s a great testbed.

Arithmetic is learnable with appropriate formatting by small cheap LLMs, but standard LLMs (trained on natural arithmetic text data) continue to not implement true arithmetic, even at the capability frontier like GPT-4, and arithmetic problems are easy to generate, benchmark, understand, and even do neural net interpretability on; so one could pilot catapulting on a pretrained LLM by looking for training schedules which make it find true arithmetic much faster than standard finetuning does.

More specifically, one would filter for ‘hard’ arithmetic problems and then search for catapult training recipes which reduce the exponent in the scaling law compared to the ‘standard’ training recipe. If one used regular arithmetic problems, the gain on rare hard problems—the sort which expose the fact that the LLM has only learned a collection of partial heuristics, approximations, and memorized answers—would be hopelessly masked by the average case.

(It is entirely possible that before perfect arithmetic performance generalization, a catapulted LLM, which has mostly succeeded in learning true arithmetic, would be outperformed on average by the regular LLM which has memorized as much as possible.)

So one would do something like filtering stringently for the 0.1% (since arithmetic is so big) of the hardest arithmetic problems (as evaluated by an existing LLM or by testing for generalization past n digits), and then use that as the heldout data that one runs scaling law sweeps on for all training recipes.

The scaling laws would ignore the average-case performance from the training runs, and also the constant factor on the hard data, and look for changes in the exponent of the scaling laws for the hard data.

Ideally, one would find something like a training recipe where after many epochs, the catapulted small LLMs are improving more rapidly on the hard data than the standard LLMs are, and that even if the catapulted LLMs are substantially worse everywhere, the more rapid improvement means that at some point “the curves cross”, and the catapulted LLMs are superior.

This would then be proof of concept that catapulting is not merely possible on a more complex problem like arithmetic that continues to challenge even SOTA LLMs, but that it changes the scaling laws.

Then to verify this result, one could apply interpretability research to it (like Zhong et al 2023): the final catapulted LLM should clearly express a valid arithmetic algorithm where the standard LLM fails to, and there should be phase transitions across the catapulted LLM’s checkpoints from standard LLM-like pseudo-arithmetics to true arithmetic.

With this proof of concept in hand, one can work on further optimizing the catapult training, and start attempting to infer what catapult training methods might scale up to SOTA LLMs like a GPT-4.

### MLPs

Why might MLPs be especially suited here? Because extreme regularization may help fix their persistent overfitting problems and provide superior scaling.

Use of sparsity like mixture-of-experts would tend to reduce the effective parameter-size and the connectivity of the NN landscape, and so would be somewhat risky. However an intriguing possibility is that catapulting might make fully-connected MLP architectures viable.

MLP architectures are much simpler, more general, parameter-efficient, more hardware-friendly than CNNs/Transformers, and look like the logical next candidate for Bitter-Lesson-ing the status quo NN architecture—but MLPs still fall far behind. Why?

The main reason seems to be that they are too powerful, and overfit. (They are to Transformers/CNNs as those architectures are to humans, one might say.) Zhao et al 2021 & Bachmann et al 2023 demonstrate that MLPs scale well and can be competitive if enough regularization (like bottleneck layers) is added.

In particular, the more regularization (and consequently, generalization downstream), the more they learn sensible convolution-like features, rather extremely high-frequency & non-robust features.

However, it is still unclear what “regularization” would preserve all the MLP benefits without crippling the architecture—and catapulting fits the bill! The high LR & catapult trajectory would suppress those MLP pathologies the same way it suppresses the milder versions in Transformers. (See Liu et al 2022, and Fan et al 2024 on how narrow deep MLPs seem to grok in unusual & better ways; likewise, He et al 2024 on deeper Transformers.)

### Hardware

A serial catapulting regime would render extremely large GPU-clusters much less useful, as they will not be able to step through each iteration much faster; it would instead place a high premium on more exotic hardware like Cerebras chips, which can execute a training step in a small fraction of the time, and hence, wallclock.

#### Expanded transclusion: Hardware Hedging Scaling Risks

Source: <https://gwern.net/blog/2024/hardware-hedging>

Hyperscalers are investing heavily in AMD/Nvidia-style GPUs optimized for moderate-scale parallelism: less than almost-shared-nothing scientific computing tasks like SETI@home, but not strictly sequential like highly-branching tasks, and with the best interconnects money can buy in a custom datacenter, probably topping out at somewhere ~1m GPUs before the communication overhead/latency & Amdahl’s law pushes the diminishing returns to 0.

If you are going to spend $50b+ on GPU hardware (and then another $50b+ on everything wrapped around them), you are going to want to invest a lot into making conservative design choices & derisking as much as possible.

So a good question here is: even if that 1m mega-GPU datacenter pencils out now as optimal to train the next SOTA, will it stay optimal? How can the hyperscalers hedge against scaling regime shifts?

---

Everyone is discussing a transition to a ‘search regime’, where training begins to consist mostly of some sort of LLM-based search. This could happen tomorrow, or it could not happen anywhere in the foreseeable future—we just don’t know. Search usually parallelizes well, and often can be made shared-nothing if you can split off multiple sub-trees which don’t need to interact and which are of equal expected value of computation.

How bad is the search regime for current hardware investments? Not too bad. In this scenario, where you are training LLMs on eg. outputs from transcripts generated by an AlphaZero-ish tree-search approach, the mega-GPU datacenter approach is fine. You can train across many datacenters in this scenario or in fact the entire consumer Internet (like Leela Zero or Stockfish do), but while maybe you wouldn’t’ve built the mega-GPU datacenter in that case, it’s as equivalent or somewhat better than what you would have, and so maybe you wound up paying 20% more to put it all into one mega-GPU datacenter (at current premiums), but no big deal. So while there are many negative economic/strategic consequences of a search regime breakthrough for the hyperscalers, in terms of enabling competition from highly distributed small-timer competitors pooling compute, and AI risk consequences (models immediately scaling up to much greater intelligence if allocated more compute), it wouldn’t render your hardware investment moot.

But it is not the case that that is the only possible abrupt scaling regime shift. Instead of getting much more parallel, training could get much less parallel, and become highly serial.

It’s worth noting that this is the reason so much scientific computing neglected GPUs for a long time and focused more on interconnect throughput & latency: actually, most important scientific problems are highly serial, and deep learning is rather exceptional here—which means it may regress to a mean at some point. There could be a new second-order SGD optimizer which cannot parallelize easily across many nodes but is so sample-efficient that it wins, or it eventually finds better optima that can’t be found by regular first-order. There could be new architectures moving back towards RNN which don’t have a “parallel training mode” like Transformers, and you inherently need to move activations/gradients around nodes a ton to implement BPTT. There could be some twist on patient-teacher/grokking-like training regimes of millions or billions of inherently serial training steps on small (even n = 1) minibatches, instead of the hundreds of thousands of large minibatches which dominates LLM training now. There could be some breakthrough in active learning or dataset distillation for a curriculum learning approach: where finding/creating the optimal datapoint is much more important than training on a lot of useless random datapoints, and so larger batches quickly hit the critical batch size. (Or something else entirely, which will seem ‘obvious’ in retrospect but no one is seriously thinking about now.)

What sort of hardware do you want in the ‘serial regime’? I think it would look a lot more like traditional scientific supercomputing than the current DL mega-GPU datacenter.

It might force a return to high-end CPUs, overclocked to as high gigahertz as possible; however, it’s hard to see what sort of serial change to DL could really cause that, aside from extreme levels of fine-grained sparsity and radical changes to the underlying neural net dynamics (if still ‘neural’ in any sense).

More plausible is that it would continue to look mostly like current DL but highly serial: like synthesizing a datapoint to train on immediately & discard, or training in a grokking-like fashion. In this case, one might need very few nodes—possibly as few as 1 model instances training. This might saturate a node of a few dozen GPUs or even hundred GPUs, but then the rest of the mega-GPU datacenter sits idle: it can run low-value old models, and otherwise has nothing useful it can do for the main training run. Any attempt to help the core GPUs simply slows them down by adding in latency.

In that case, you don’t want GPUs, which are too parallel, or CPUs, which are fast at serial tasks but still way too general purpose given that you are still running a conventional neural network here. What you want is a single chip which computes forwards and backwards passes of a single model as fast as possible.

---

Groq chips don’t do training, so they are right out. What comes to mind is Cerebras: a single ungodly fast chip is exactly their premise, and was originally justified by the same rationale given above as it applies to scientific computing. Cerebras doesn’t work all that well for the current scaling regime, but in a serial scaling regime, that could change drastically—a Cerebras chip could potentially be many times faster for each serial step (regardless of its throughput) which then translates directly to an equivalent wall-clock speedup. (Kosson et al 2020 unfortunately doesn’t include useful timings for our purpose here. Cerebras’s marketing material gives an example of a linear system solver which takes ~2,000 microseconds per iteration on a CPU cluster, but only 28 microseconds on a CS-1 chip, so >200× faster per iteration; a more recent update benchmarks Llama-3.1-70b at 20× H100s & 2.4× Groq.)

The implication then is that whoever has the fast serial chips can train a model and reach market years ahead of any possible competition.

If, for example, you want to train a serial model for half a year because that is just how long it takes to shatter SOTA and optimally trades-off for various factors like opportunity cost & post-training, and your chip is only 50× faster per iteration than the best available GPU (eg. 1ms to do a forwards+backwards pass vs 50ms for an Nvidia B200), then the followers would have to train for 25 years! Obviously, that’s not going to happen.

Competitors would either have to obtain their own fast serial chips, accept possibly staggering levels of inefficiency in trying to parallelize, or just opt out of the competition entirely and go to the leader, hat in hand, begging to be the low-cost commodity provider just to get some use out of their shiny magnificently-obsolete mega-GPU datacenter.

---

Is this particularly likely? No. I’d give it <25% probability. We’ll probably just get AGI the mundane way with some very large mega-GPU datacenters and/or a search transition. But if you are spending $100b+, that seems likely enough to me to be worth hedging against to the tune of, say, >$0.1b?

---

How would you invest/hedge? Groq/Tenstorrent/AMD/Nvidia/Etched are all out for various reasons; only Cerebras immediately comes to mind as having the perfect chip for this.

Cerebras’s last valuation was apparently $4b & they are preparing for IPO, so investing in or acquiring Cerebras may be too expensive at this point. (This might still be a good idea for wealthy investors who have passed on Cerebras due to them having no clear advantage in the current regime, and haven’t considered serial regimes as a live possibility.) Investing in a startup intended at beating Cerebras is probably also too late now, even if one knew of one: it takes too long to iterate on these giant chips, to get them big enough so only a few are necessary and can achieve low latency.

What might work better is negotiating with Cerebras for options on future Cerebras hardware: Cerebras is almost certainly undervaluing the possibility of a serial regime and not investing in it (given their published research like Kosson et al 2020 focused on how to make regular large-batch training work and no publications in any of the serial regimes), and so will sell options at much less than their true option value; so you can buy options on their chips, and if the serial regime happens, just call them in and you are covered.

The most aggressive investment would be for a hyperscaler to buy Cerebras hardware now (with options negotiated to buy more hardware) to try to make it happen. If one’s researchers crack the serial regime, then one can immediately invoke the options to more intensively R&D/choke off competition, and begin negotiating an acquisition to monopolize the supply indefinitely. If someone else cracks the serial regime, then one at least has some serial hardware, which may only be a small factor slower, and one has sharply limited the downside: train the serial model yourself, biting the bullet of whatever inefficiency comes from having older / too little serial hardware, but then you get a competitive model you can deploy on your mega-GPU datacenter and you have bought yourself years of breathing room while you adapt to the new serial regime. And if neither happens, well, most insurance never pays off and your researchers may enjoy their shiny new toys and perhaps there will be some other spinoff research which actually covers the cost of the chips, so you’re hardly any worse off.

#### End expanded transclusion

Incidentally, the use of low-latency hardware would also open up more exotic neural net architectures like AUNN/IFNN.

### Prototyping With Image Classification

For non-LLMs like CNNs/MLPs trained on CIFAR-10/CIFAR-100 or ImageNet-1k (MNIST being too trivial to be convincing), we would similarly expect more human-like images. In image classification, it’s harder to define ‘hard’ than arithmetic, and the standard NN accuracy is so high that simply filtering for errors might not yield enough ‘reasonable’ errors at this point.

So we would instead use one of the many ‘post-ImageNet’ image datasets designed to stress-test classifiers, like ImageNet-A, ImageNet-C, ImageNet-Hard, ImageNet-R, & ImageNet-Sketch. These would be used as the target in scaling law sweeps as described previously.

#### Adversarial Robustness

A possible alternative would be to look at adversarial robustness instead of a standard benchmark.[^13]

If the dimpled manifold thesis is correct, then I interpret it as predicting that: while HLLMs might be intrinsically robust per isoperimetry, so too might be tiny models, as long as they found the right manifold which did not require the “dimples” (ie. unprincipled, ad hoc, memorized tweaks to the decision boundaries to get the right answer). This true manifold would potentially be found by catapulting.

So if one successfully catapulted a small CNN, even on CIFAR-10, it might demonstrate adversarial robustness and generalization (rather than the usual small NN choice of either robustness or generalization). This might be especially the case with MLPs, and while an additional research risk, the efficiency of MLPs would allow extensive testing on just 1 GPU (eg. Bachmann et al 2023).

No theory of adversarial examples other than “non-robust features” & “dimpled manifold” predicts that small models might be adversarially robust if simply trained for a long time with an odd learning rate schedule, so any large improvement in adversarial robustness is an important finding.

### Prior Art

There is essentially no research on training >10t-parameter LLMs, or cyclical LRs on large LLMs (as opposed to small GPT-2-scale ones).

Historically, this is due to a mix of academic fashion/prejudices and capability gains by smaller models. Work on training highly-overparameterized LLMs, or on the equiparameterization regime[^14], was largely killed by the release of the Chinchilla paper, which provided the perfect excuse for everyone to immediately halt parameter-scaling (since it no longer led immediately to SOTA results), as they have always wanted to, for a mix of good and bad reasons.

Similarly, the excuse of “optimizing for inference-optimality” by overtraining small models has become popular, by optimizing scaling laws which assume that the trained model will be naively deployed as-is, without the actual pruning, quantizing, & distilling everyone does anyway.

This means that the field of high-energy DL is wide-open: this proposal will be highly unpopular, it is much less likely than usual that anyone would independently investigate this direction, and they will be discouraged by poor preliminary results when the training runs appear to have simply failed (because of subtle bugs, poor hyperparameters, or simply inadequate training time to catapult into a region where better results can be benchmarked—assuming the right benchmarks are being used to begin with).

### Benchmarking

This is such a different training regime that previous scaling law sweeps are inapplicable. Further, the goal here may be one that existing benchmarks actively mislead on. They test mostly common easy questions—the sort where ‘direct fit’-like thinking does best on, by definition.

So a major question here would be whether scaling laws should target perplexity as usual, or if they need to target a custom benchmark which tries to test human-like generalization rather than memorization.

Given the difficulty we have in constructing non-trivia-heavy benchmarks which existing LLMs can’t beat, this might be one of the hardest parts! But I suspect that after training a reasonable HLLM, possibly via trial-and-error, and interacting with it for a while to get an idea of how it acts qualitatively, the right metric might become more obvious.

The benchmarks might include adversarial robustness, hard negative mining (ie. the hardest problems that the best LLMs still get wrong), meta-learning, checking how much sets of models add to an ensemble’s accuracy[^15], or use a metric which rewards performance while penalizing memorization of training data.

### Capabilities

The final model should generalize much better—possibly achieving the Nyquist learner limit of perfectly modeling the true (non-dimpled?) latent manifold, and thereby constructively answering Rosenfeld’s question about how a band-limited Nyquist learner could be implemented in current NN architectures.

Per the lottery ticket hypothesis, once the true generalizing algorithm has been found, and has been further trained as desired (perhaps on some large trivia-heavy corpus), we can prune it down to a much smaller, faster, more feasible-to-use model, in an example of “train large, then compress”.

Given the capacity of even small models, it may be possible to finetune these smaller models on arbitrarily large amounts of data to exploit the scaling laws of transfer. Because they start with the human-like generalizing prior, they will learn & memorize the data as appropriate, but without the pathologies of ‘direct fit’ to the data as when trained from scratch on the same data—thereby achieving the best of both worlds.

#### Economic Implications

If catapulting LLMs is enough to close the gaps with humans and solve AGI, then their economics simply turn into a discussion of AGI; but what if they are much better, yet still far from AGI?

As of 2024, the economics of the largest SOTA LLMs are poor, because model capabilities can be so easily cloned by using the cheap APIs to create large training corpuses, enabling behavior-cloning of the superior models. This makes it possible to ‘clone’ the most expensive multi-billion-dollar LLM into a cheap or even FLOSS LLMs (justified by commoditize-your-complement dynamics), eroding margins within months of release—even if the cloned model is broadly inferior.

AI scaling companies are trapped in a race to spend the most capital on ever-larger training runs & deploy dirt-cheap distilled versions to gain market-share before the cloners erode their edge, in the hope that they can achieve some network effect or ‘escape velocity’ (like creating AGI).

For a catapulted LLM, however, this would still be the case, but much more so. For catch-up players, catapulting eliminates data-constraints, but the compute-cost (and difficulty) of training a catapulted LLM may be insuperable: the training run itself might not be too expensive, but the trial-and-error and proprietary trade-secret knowledge of the special sauce of how exactly to make catapulting work may be extremely compute-expensive.

And they can continue to do the cloning as usual, but whereas in 2024, the cloned models are not that much inferior and share all the same weaknesses as the SOTA model they are cloning (suffering from adversarial examples, confabulations, bizarre simple errors and brittleness etc.), in a catapult scaling regime, this would not be the case: there would be a clear qualitative difference.

In fact, the cloned models may even be superior on many benchmarks, because they are trained so heavily the ‘standard’ low-learning-rate way and enjoy all the benefits of that approach, but still disfavored by users, who can’t afford their unreliability, confabulations, and blandness.

However, the catapulters remain able to do all the usual tricks and can create their own superior ‘clone’ models.

(And these clone models can themselves presumably be catapulted for regimes or tasks where there is insufficient data to train the low-learning-rate direct-fit way.) Given the economics of scaling DL, where the compute-cost can be dropped by extremely large amounts while amortizing the initial training cost over many users who provide high usage of GPUs, this further means the first-mover in catapulting potentially can drop prices enough to discourage competition.

So a catapult LLM creator, if the catapulted LLM has enough of a human-like edge in reliability & quality, may be able to maintain high margins much longer than 2024 LLMs do.

### Alignment

Speculating further, on the premise of the above: most capability improvements do not help AI alignment as much. This is because they either are specific to a capability in a narrow domain, or they enhance broad capabilities but only in a ‘brittle’ non-generalizing way, which can be highly economically valuable but doesn’t help ‘alignment’ much because we are interested in alignment not for its current narrow in-domain but generalizing.

We need the kind of alignment which doesn’t help a chatbot say socially-acceptable things today (or do the right things for the wrong reasons), but which would make chatbots in charge of civilization do socially-acceptable things in the future (and do the right things, for the right reasons, so they will keep on doing the right things indefinitely).

But this catapult or Nyquist learner may be the exception, because what it helps with is true generalization.

A catapulted LLM trained on large amounts of morality-related text has not learned purely an assemblage of memorized fragments, heuristics, tricks, and statistical associations, with any underlying algorithm begrudgingly forced by scaling, or learned deception & situated reasoning to maximize a reward, or been adversarially selected by use of ‘interpretability’ techniques to learn to think in nonlinear opaque ways which do not raise any red flags; instead, it has learned the underlying value-manifold the hard way, like a highly-intelligent, grown, moral adult human has.

Even if the capability improvement turns out to be beaten by standard LLM scaling approaches (like simply brute-force annotating every error), true generalization would be invaluable to alignment.

It does not solve the alignment problem in generality—but it might provide a way to create a genuinely moral AI, and that is a good starting point.

### Interpretability

Further, because the native neural net way of thinking is a large complicated pastiche of memorization & heuristics, while the overparameterized grokked LLM has distilled out an algorithmic core for the key tasks, such a catapulted LLM ought to be much more useful for interpretability work. We can more easily validate that they are genuinely moral AI.

Once the overparameterization has been removed, what is left should be natively much closer to simple, verifiable, interpretable, and extractable algorithms. These can then be formally analyzed & verified (possibly with the assistance of the genuinely-moral AI LLMs, which are still risky but not too risky if they are probably moral to begin with, and we confine them to carefully-checked formal outputs and tasks like algorithmic equivalence of a sparsified neural network and extracted sub-algorithms.)

## External Links

- Discussion: HN

## Appendix

### Dynamic Grokking

> An alternative to grokking of full models might be grokking of specific problems. Pretraining might be the cheap but incomplete way of training a brain, followed by invoking a much more expensive “pondering” process which spot-fixes important errors.
>
> How would you grok a specific problem in an LLM? You might do it by training it repeatedly on the problem, finetuning the model on a very small data sample (ie. dynamic evaluation).
>
> Done repeatedly, this could lead the model to explore deeply in the loss landscape, eventually reaching new basins, generating qualitatively different outputs and even having ‘creative breakthroughs’.

Inducing grokking on a pretraining corpus may be neither desirable nor possible because it would require too much training on all problems. However, we may still want to grok on some problems—particularly hard ones.

At present, there is not really any way to take a hard problem that an LLM fails to solve, and let the LLM “think about it”. If a bunch of inner-monologue completions do not solve it, there’s no straightforward way to “ponder”. You cannot spend a lot of money or compute on an important problem to solve it; even GPT-4 o1 appears to have a hard limit, set by its context window size.

There is currently no AI equivalent of a human thinking for a long time, seemingly getting nowhere, but then (incubation effect) having an insight and solving it (or at least making useful progress). Grokking, however, looks an awful lot like “a neural net thinking about a problem for a very long time until it starts to understand it and then starts to have insights & solve it”.

What would it look like to try to have an incubation effect in AI?

#### Pondering ≠ Tree Search

Usually, such ‘search’ is imagined as a kind of MCTS-esque tree search, where the human brain runs a deep tree search ‘in the background’ and eventually it finds a solution and that bursts into consciousness as a eureka moment or L’esprit de l’escalier.

But are humans running some sort of MCTS-esque tree search? I have my doubts.

A tree search requires expanding many nodes and building a large deep tree—where is this all happening in the human brain, with its highly-limited working memory? If it’s so deep and requires days or months of search, where is it stored all that time while being constantly updated & traversed, while being robust to all the other activities we do, like sleep? And why does this sort of thinking seem to benefit so much from sleep, even short periods? The kind of hippocampal ‘experience replay’ we know the brain is doing has a highly spatial dimension and is repetitive, and so doesn’t seem too much like we’d expect from a tree search.

Why does it seem to require a consistent level of effort over a long time, often with no progress and just beating one’s head against the wall as one goes around in intellectual circles? If we are so good at it that it can be totally unconscious, why do we find conscious tree search, like thinking through lines of play in chess, so difficult? Why can’t we simply run these searches in the background indefinitely, stop thinking about a problem entirely, and then years later be startled by sudden Nobel-Prize-worthy insights out of the blue?

What this looks like to me is closer to an effect of neuroplasticity: the constant repeated thoughts force neuroplasticity and neurological changes, which are then heavily regularized during sleep; if not done often enough, the benefits disappear, but if done often enough, despite no apparent change in the outputs, the brain keeps changing as it tries to find the optimal ‘solution’ to compressing/predicting the problem inputs (and all intermediate results of value); eventually the brain has updated enough times to move to a new basin and experiences a phase shift in the solutions it has been futilely trying, and comes up with a valuable novel output, which suddenly appears in consciousness without any apparent cause—because the cause is the low-level neuron changes which are totally inaccessible to conscious introspection.

#### Neuroplasticity = Dynamic Evaluation

What would be the closest thing here in LLMs? Neuroplasticity is equivalent to finetuning, or dynamic evaluation—finetuning a model at runtime on the new inputs, which can greatly boost LLM performance.

It has been observed that self-attention is equivalent to gradient descent: it updates the model at runtime by doing a constricted form of gradient descent encoded into the forward pass as “fast” weights computed by the slow weights per-problem. Enough self-attention is equivalent to gradient descent on the same data… but it becomes increasingly inefficient. Dynamic evaluation is similar, and hits a middle point between self-attention and fullblown training.

Notably, there is a 3-way tradeoff between model scale, context window scale, and use of dynamic evaluation: dynamic evaluation can boost a small model up to the equivalent of a model which is much larger or where there is much more relevant data in a sufficiently-large context window.[^16]

Dynamic evaluation works both for supervised learning like image classification, but it was introduced in RNNs for unsupervised learning by next-token prediction, and does not require labels or known-answers: eg. you can do dynamic evaluation on the questions in a dataset, to improve prediction of the still-unknown answers.

Using dynamic evaluation lets a model learn and think more deeply about inputs, by letting it do multiple implicit gradient descent steps to update itself per problem: the self-attention tries to analyze the inputs, and then their results are distilled into the weights by the dynamic evaluation, which improves the next self-attention pass, and so on—the learning done during the self-attention gets amortized into the model weights, instead of being thrown away only to be re-computed again, taking up time during the next forward pass.

Dynamic evaluation is usually done as a single-pass for convenience & efficiency, but there is nothing inherently requiring it, and for difficult inputs, you might want to do more than one, and run multiple steps of dynamic evaluation. This is because two gradient descent steps are not necessarily the same as one step with a larger learning rate, and the more iterative “looks” at the data, the more the model can “change its mind” in a way that a single step can’t.

(An analogy would be experience replay, where a RL model can learn from a datapoint dozens of times, or expert iteration like AlphaZero: the model and the search bootstrap each other up, which is highly non-stationary, where one alone would simply sigmoid hard.)

One might then think that this can’t be useful because given the sample-efficiency of LLMs, one or two steps is probably enough to memorize the current input, and that one risks catastrophic forgetting… but that is an intuition which seems to be wrong in practice—LLMs memorize many things, and yet generalize, catastrophic forgetting just doesn’t happen with large pretrained models due to their extreme overparameterization & sample-efficiency enabling continual learning, and grokking & related phenomena show that there is still benefit to training long after it would appear to be useless.

#### Repeated Neuroplasticity As Implicit Search

So if we put this together, we can imagine using dynamic evaluation to try to trigger a kind of pondering which avoids tree search and can potentially lead to grokking: call it dynamic grokking.

In dynamic grokking, we spend a lot of compute to try to solve a specific hard problem. We repeatedly do dynamic evaluation on the input to update the model, corresponding to a “learning” neuroplasticity step, and roll out a new completion, corresponding to a “pondering” step. If the completion is correct, or we run out of compute, we stop; otherwise, we repeat.

During dynamic grokking, the LLM will update possibly thousands of times, immediately memorizing the input but hopefully gradually changing deep within, eventually generating novel completions and solutions that the original unimproved ‘base’ model could not have generated even with tens of thousands of brute-forced completions. These final results can then be given to the user, used as the seed for other search techniques or a new dynamic grokking, trained on to improve base models, etc.

Extensions to this would include: periodically doing dynamic evaluation but with heavy weight decay regularization, corresponding to a “sleep” step; updating the input with the best completions or parts of a completion (possibly using a second prompt to generate a useful summary of one completion); attempting to systematically vary the context to produce less redundant updates (eg. dropping out parts of the input, retrieving different documents from a database, injecting noise into the forward passes); and saving the model snapshots to merge together into a better base model (which aggregated over many problems, may steer the model towards a MAML-esque meta-learning model which learns how to dynamically learn). Given the repeated iteration of training, this would benefit from hardware setups focused on minimizing latency, like Cerebras.

(It is possible that a small model using dynamic grokking on Cerebras-like hardware might be able to outperform time-to-solution of even a highly-parallelized but static large model, depending on how poorly static models scale their searches.)

As far as I am aware, there is no prior art relevant to dynamic grokking, because dynamic evaluation is almost completely unused, and the few inner-monologue approaches which do gradient descent at all will usually only do it with something like a subset of completions filtered for quality.

While dynamic grokking can be researched or tested with the usual toy problems and approaches, it’s unclear how you could, even in principle, get an idea if dynamic grokking is working on a specific problem. The externally-observable outputs (tokens or logits) might not change at all for many iterations; and all the ‘internal’ metrics for grokking I’ve seen are unscalable, architecture/problem-specific, noisy, or otherwise flawed.

## Notes

[^1]: A personal example: my mother tells me that when we got our first Mario Brothers video game, I spent weeks playing it by running Mario to the first pit and deliberately jumping into it just to hear the sound effects of Mario dying (which I apparently found hysterically funny), and that it drove her crazy. What could my brain have possibly learned from the 5,000th repetition of the sound effect? And if I wasted weeks on learning nothing, then how did I manage to be even more sample-efficient in the rest of my life?

[^2]: Similarly, children do not seem to learn languages all that more efficiently than adults and have some special sample-efficiency; an adult who is serious (ie. spends hours a day studying using total immersion & spaced repetition, not pretending to learn using Duolingo) will not take 18 years to reach native proficiency. Like learning new systems, the reason children ‘beat’ adults seems to come down mostly to a willingness to do the work—to be embarrassed, to fail, and to just spend as many hours as it takes obsessing over the new thing rather than resting on one’s laurels.

[^3]: When we dare to project out scaling laws all the way to human or superhuman performance, they generally do not require absurd amounts like quadrillions of parameters, which would appear to be implied by most of the biological-supremacy views.

[^4]: Aside from not making use of non-robust features, despite their power & omnipresence, humans appear to be almost completely blind to them.

[^5]: We do not simply scale up models to the largest one that fits in our hardware because they cost more compute per-step, so a compute-optimal small model will process much more data with the same compute budget and eventually beat them.

[^6]: This is much less remarked on in the savant literature. Is confabulation under-documented because no one expects it, and so ignores it or fails to test for it (eg. by asking trick questions)? Or do savants refuse to answer such questions, if it is not in one of their special interests or if they feel confused when they try to answer it? (Down also notes that savants are not always cooperative, and one can imagine that it might take a long period of building rapport to reach the point where one could ask anything, instead of the usual displays of skill which they may take great pride in.)

[^7]: Synesthesia: S.’s case, as many readers have noted, resembles the Jorge Luis Borges story “Funes the Memorious”, a fictional work about a man plagued by the persistence of his memory. “To think is to forget a difference, to generalize, to abstract”, Borges writes. “In the overly replete world of Funes there were nothing but details, almost contiguous details.” Similarly, Luria writes that for S. almost every word, every thought, was freighted with excessive detail. When he heard “restaurant”, for example, he would picture an entrance, customers, a Romanian orchestra tuning up to play for them, and so on. Like Funes, S. had a sort of private language to catalogue the richness of his mental associations. The word for “roach” in Yiddish could also mean, in his mind, a dent in a metal chamber pot, a crust of black bread, and the light cast by a lamp that fails to push back all the darkness in a room.

[^8]: In fact, this sort of delayed or ‘deceptive’ payoff is an optimization task that evolutionary strategies are competitive with gradient-based methods, because they optimize the net fitness over a lifetime; it does not matter to evolution how useless an infant looks on benchmarks (yielding a deceptive gradient) if the converged adult has higher fitness.

[^9]: In the terminology of Huang et al 2024, we normally train models in the “progression”/“semi-grokking” regimes (underfitting), but by increasing the model size, we can move it to the “grokking” regime, where it may memorize but then generalizes much better than progressing models; and if we filter the data stringently enough to minimize the ability to memorize, this ‘ratio’ will move the crossover point much earlier in training (see also Huang et al 2024’s observation about how adding a pure-memorization task “poisons” grokking, which parallels Dohmatob et al 2024 about noisy/low-quality synthetic data).

[^10]: Such large networks have been demonstrated to run on GPU clusters, although they have all been mixture-of-experts or recommender models.

[^11]: Whether RNNs can grok/catapult, and if that could help them leapfrog Transformers, is an interesting question that I don’t believe has been tried much, given the focus on fully-connected/CNN/Transformers.

[^12]: Or would it make more sense to instead have a constant large learning rate, and instead cycle the weight decay, to let weights grow and then brutally prune them back, to increase the analogy to “sleep”? Smith 2022 claims that a high-learning-rate+cyclical-WD schedule works & may increase sample-efficiency.

[^13]: One could try to test this with LLMs too, but adversarial examples are not as well understood or easy with LLMs compared to the standard CNN/ViT image classification adversarial examples, so would not be as good for prototyping.

[^14]: One of the reasons to favor overparameterized models is that an extreme level of parameterization (compared to Chinchilla) keeps showing up in scaling theory papers as optimal, and in particular, a 1:1 equiparameterization ratio—one datapoint, one feature. This seems especially intuitive if datapoints are optimally selected (or synthesized), and can in theory achieve much more favorable scaling like exponential decrease in error.

[^15]: To the extent that normal models are all trapped in the same nearby basins, they will make the same sorts of errors and ensembling will not improve metrics much. To the extent that they are exploring the full loss landscape and finding far away models, they should have much less correlated errors and so ensemble better than expected.

[^16]: Note that dynamic evaluation is better than the alternatives in the sense that we can always do dynamic evaluation on a model we are using, whereas we usually cannot use a larger model nor put more into the context window. (And if we could, then we could do dynamic evaluation on that one instead.)
