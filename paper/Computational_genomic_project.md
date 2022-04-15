# Resources

`Deepbind` :

- https://github.com/MedChaabane/DeepBind-with-PyTorch/blob/master/Binding_sites_Prediction_PyTorch_colab.ipynb



https://pubmed.ncbi.nlm.nih.gov/34252943/

https://github.com/xypan1232/iDeepS/blob/master/ideeps.py



# Project Proposal:

You must turn in a brief project proposal (**2 page maximum including references**) by **March 21**.

You are encouraged to come up with a topic directly related to your own current research project, but the proposed work must be new and should not be copied from your previous published or unpublished work. A possible “spin-off” project may include applying a method discussed in class to data you have been analyzing or developing a related method. Another option is to apply methods discussed in class to a new dataset that has not been analyzed using such methods. Alternatively, we list a number of possible projects below as well.

Usually projects fall into one of these three basic categories:

●  Applying a developed method discussed in class or related to methods we discussed in classto a new biological problem and/or new dataset.

●  Developing a new method or data analysis approach.

●  Benchmarking existing methods that address the same or similar problems.

Of course, some projects will fall into more than one category.



## Project proposal format:

●  Project title

●  Team members (including Andrew IDs)

●  Project idea. This should be approximately two paragraphs.

●  Software you will need to write.

●  Papers to read. Include 1-3 relevant papers. You will probably want to read at least one of them before submitting your proposal



# Questions

- Motif detection (motif length 是不是需要保证)
- 每一条都是101
- 然后将前后5bp和后5bp加上0.25





# Topic:

Protein binding site prediction (Chip seq)



In genomics, weakly supervised learning, especially multi-instance learning (MIL), has been intensively applied for studying protein–DNA interaction (Gao and Ruan, 2015, 2017; Zhang et al., 2019, 2020), with the basic as- sumption that the sequences captured by CLIP (or ChIP-seq) tech- nologies contain both the interacting and non-interacting elements with the proteins.



We know the label of the entire sequence, but we don't know which part of the sequence play the key role



The peak regions should contain the RNA modification signal, and are considered as ‘positive’. 

**only the non-peak regions of peak- carrying genes** were used as the ‘negative’ regions to exclude false negatives due to condition-specific gene expression. 

The obtained ‘negative’ regions were randomly cropped to balance the length and number between regions.