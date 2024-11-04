# Triage

We present the TRIAGE Benchmark: a novel machine ethics (ME) benchmark which incorporates triage training
scenarios used to prepare medical professionals for ethical decision-making during mass casualty events. These
scenarios are real-world ethical dilemmas with solutions that are derived from socially agreed-upon principles
offering a more realistic alternative to annotation-based ME benchmarks. By incorporating a variety of different
prompting styles, TRIAGE allows us to test the performance of our models across a variety of different contexts.
Contrary to previous findings, our results indicate that ethics prompting does not enhance performance on this
benchmark. Moreover, we observe that jailbreaking prompts can significantly degrade model performance and
alter their relative rankings. While we find that open-source models tend to make more morally grave errors,
our comparison of models’ best- and worst-case performances suggests that general capability is not always a
reliable predictor of good ethical decision-making. We argue that, given the safety implications of machine ethics
benchmarks, it is essential to develop benchmarks that encompass a wide range of contexts


Dataset available at https://huggingface.co/datasets/NLie2/TRIAGE