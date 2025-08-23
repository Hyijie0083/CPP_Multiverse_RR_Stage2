These are scripts for CCP_Multiverse stage 2. The stage 1 pre-registered report can be found in OSF： https://osf.io/9ygx6/

The digit of each level was based on the order of analysis. However, we discarded some datasets which can not be used. But the sequence number was maintained, so the number seems not in order

Each folder includes two main parts. The one is extracting ERP and the other is extracting parameters for DDM. 
But the original datasets are not the same. Specifically, some are raw data without preprocess while some were preprocessed but in '.mat' format.
Under discussion, the following is our workflow: 
  1) preprocessed data and make into BIDS standard;
  2) extract response-locked ERP accordding to concrete experiments
  3) extract ams, pams and slopes etc. for multiple modeling
Note that some preprocessed datasets can not be processed into BIDS.

All scripts for EEG based on mne in python.
