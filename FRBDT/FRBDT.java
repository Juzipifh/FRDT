package keel.Algorithms.Decision_Trees.FRBDT;

import keel.Algorithms.Decision_Trees.FRBDT.core.*;

import keel.Dataset.Attribute;
import keel.Dataset.Attributes;
import keel.Dataset.InstanceSet;

 public class FRBDT  extends Classifier implements WeightedInstancesHandler {
    static final long serialVersionUID = -6589312896832147161L;

	/** The limit of description length surplus in ruleset generation */
	private static double MAX_DL_SURPLUS = 64.0;

	/** The class attribute of the data*/
	protected AttributeWeka m_Class; 

	/** The ruleset */
	public FastVector m_Ruleset;

	/** Runs of optimizations */
	private int m_Optimizations;

	/** # of all the possible conditions in a rule */
	protected double m_Total = 0;

	/** Whether in a debug mode */
	protected boolean m_Debug = false;

	/** The class distribution of the training data*/
	double[] aprioriDistribution;

    boolean m_useRuleStretching=false;

    public double m_alpha=0.02;

    public int m_maxAttUsed=5;
  
    public double m_shreshold=0.6; 

    /**
	 * Constructor.
	 *
	 * @param params The parameters of the algorithm
	 */
	private String trainFile, evalFile, testFile;
	private String outputTrainFile, outputTestFile, outputClassifierFile;
	/** filter: Normalize training data */
	public static final int FILTER_NORMALIZE = 0;
	/** filter: Standardize training data */
	public static final int FILTER_STANDARDIZE = 1;
	/** filter: No normalization/standardization */
	public static final int FILTER_NONE = 2;

    public FRBDT(parseParameters parameters) {

		m_maxAttUsed = Integer.parseInt(parameters.getParameter(1));//2
		m_shreshold = Double.parseDouble(parameters.getParameter(2));//2
		m_alpha=Double.parseDouble(parameters.getParameter(3));//2


		trainFile = parameters.getTrainingInputFile();
		evalFile = parameters.getValidationInputFile();
		testFile = parameters.getTestInputFile();
		outputTrainFile = parameters.getTrainingOutputFile();
		outputTestFile = parameters.getTestOutputFile();
		outputClassifierFile = parameters.getOutputFile(0);

	}

    /**
	 * Creates a new allocated WEKA's set of Instances (i.e. Instances) from a KEEL's set of instances
	 * (i.e. InstanceSet).
	 * @param is The KEEL Instance set
	 * @param preprocessType An integer with the type of preprocess done before exporting data to Weka format (0 = normalize, 1 = standardize, 2 = do nothing).
	 * @return A new allocated WEKA formatted Instance set
	 */
	protected Instances InstancesKEEL2Weka(InstanceSet is, int preprocessType) {
		Attribute a, newAt;
		Instance instW;
		keel.Dataset.Instance instK;
		int out, in, newNumAttributes, enlargedValueVectorPos;
		double values[];
		Instances data;
		FastVector atts;

		// Create header of instances object
		out = Attributes.getInputNumAttributes(); //the class attribute is usually the last one
		atts = new FastVector(Attributes.getNumAttributes());
		for (int i = 0; i < Attributes.getNumAttributes(); i++) {
			a = Attributes.getAttribute(i);
			//atts.addElement(a);
			AttributeWeka aWeka;
			if (a.getType() == a.NOMINAL) {
				FastVector nominalValues = new FastVector(a.getNumNominalValues());
				for (int j = 0; j < a.getNumNominalValues(); j++) {
					nominalValues.addElement(a.getNominalValue(j));
				}
				aWeka = new AttributeWeka(a.getName(), nominalValues, i);
			} else {
				String range = new String("[");
				range += a.getMinAttribute();
				range += ",";
				range += a.getMaxAttribute();
				range += ")";
				aWeka = new AttributeWeka(a.getName(), i);
				aWeka.setNumericRange(range);
			}
			atts.addElement(aWeka);
			if (a.getDirectionAttribute() == Attribute.OUTPUT) {
				out = i;
			}
		}
		data = new Instances(Attributes.getRelationName(), atts,
				is.getNumInstances());
		data.setClassIndex(out);
		newNumAttributes = Attributes.getNumAttributes();

		//now fill the data in the data instanceset
		for (int i = 0; i < is.getNumInstances(); i++) {
			instK = is.getInstance(i);
			in = out = 0;
			enlargedValueVectorPos = 0;
			values = new double[newNumAttributes];
			for (int j = 0; j < Attributes.getNumAttributes(); j++) {
				a = Attributes.getAttribute(j);
				if (a.getDirectionAttribute() == Attribute.INPUT) {
					if (a.getType() != Attribute.NOMINAL) {
						values[enlargedValueVectorPos] = instK.
						getAllInputValues()[in];
						enlargedValueVectorPos++;
					} else {
						values[enlargedValueVectorPos] = instK.
						getAllInputValues()[in];
						enlargedValueVectorPos++;
					}
					in++;
				} else {
					values[enlargedValueVectorPos] = instK.getAllOutputValues()[
					                                                            out];
					out++;
					enlargedValueVectorPos++;
				}
			}
			//**IMPORTANT** We set the weight of the instance to ONE
			instW = new Instance(1, values);
			data.add(instW);
		}

		return data;
	}
    	/**
	 * It launches the FRBDT algorithm
	 */
	public void execute() {
		Instances isWeka;
		Instance instWeka;
		InstanceSet IS = new InstanceSet();
		InstanceSet ISval = new InstanceSet();
		InstanceSet IStest = new InstanceSet();

		try {
			//*********build the FR3 classifier********/
			IS.readSet(trainFile, true);
			isWeka = InstancesKEEL2Weka(IS, FILTER_NONE);
			buildClassifier(isWeka);
			double featureSum=0.0;
			int ruleNum=0;
			for(int i =0;i<m_Ruleset.size();i++){
				FastVector rules =(FastVector)m_Ruleset.elementAt(i);
				ruleNum+=rules.size();
				for (int j=0;j<rules.size();j++){
					RipperRule rule=(RipperRule)rules.elementAt(j);
					featureSum+=rule.size();
				}
			}
			System.out.println("Number of rules:"+ruleNum+"\n");
			System.out.println("Number of features:"+1.0*featureSum/ruleNum+"\n");

			//********validate the obtained FR3*******//

			ISval.readSet(evalFile, false);
			isWeka = InstancesKEEL2Weka(ISval, FILTER_NONE);
			// obtain the predicted class for each train instance

			Attribute a = Attributes.getOutputAttribute(0);
			String outputVal = new String("");
			int hits = 0;
			for (int i = 0; i < isWeka.numInstances(); i++) {
				keel.Dataset.Instance inst = ISval.getInstance(i);
				instWeka = isWeka.instance(i);
				instWeka.setDataset(isWeka);
				int outputClass = (int)this.classifyInstance(instWeka);
				String realClass = inst.getOutputNominalValues(0);
				String predictedClass = a.getNominalValue(outputClass);
				if (realClass.compareTo(predictedClass) == 0) {
					hits++;
				}
				outputVal += realClass + " " + predictedClass+"\n";
			}
			double accTrain = 1.0 * hits / isWeka.numInstances();

			IStest.readSet(testFile, false);
			isWeka = InstancesKEEL2Weka(IStest, FILTER_NONE);
			String outputTest = new String("");
			hits = 0;
			for (int i = 0; i < isWeka.numInstances(); i++) {
				keel.Dataset.Instance inst = IStest.getInstance(i);
				instWeka = isWeka.instance(i);
				instWeka.setDataset(isWeka);
				int outputClass = (int)this.classifyInstance(instWeka);
				String realClass = inst.getOutputNominalValues(0);
				String predictedClass = a.getNominalValue(outputClass);
				if (realClass.compareTo(predictedClass) == 0) {
					hits++;
				}
				outputTest += realClass + " " + predictedClass+"\n";
			}
			double accTest = 1.0 * hits / isWeka.numInstances();
			writeOutput(outputVal,outputTest,accTrain,accTest);

		} catch (Exception ex) {
			System.err.println("Fatal Error building the FRBDT model!");
			ex.printStackTrace();
		}
		;

		Files.writeFile(outputClassifierFile, toString() + "\n\n\n\n" + "REGLAS = " + m_Ruleset.size());
	}

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        /** for serialization */
        instances = new Instances(instances);
		instances.deleteWithMissingClass();

		aprioriDistribution = new double[instances.classAttribute().numValues()];// the instances number of each class
		boolean allWeightsAreOne = true;
		for (int i = 0 ; i < instances.numInstances(); i++){
			aprioriDistribution[(int)instances.instance(i).classValue()]+=instances.instance(i).weight();
			if (allWeightsAreOne && instances.instance(i).weight() != 1.0){
				allWeightsAreOne = false;
				break;
			}
		}
        Instances data = new Instances(instances);

		if(data == null)
			throw new Exception(" Unable to randomize the class orders.");

		m_Class = data.classAttribute();	
		m_Ruleset = new FastVector();
        if(m_Debug){
			System.err.println("Sorted classes:");
			for(int x=0; x < m_Class.numValues(); x++)
				System.err.println(x+": "+m_Class.value(x) + " has " +
						aprioriDistribution[x] + " instances.");
		}
        boolean stop =false;
        while(!stop){
            Instances nextLayerData=rulesetForOneLayer(data);
            stop=checkStop(data, nextLayerData);
			data=nextLayerData;
        }
    }

    protected boolean checkStop(Instances data, Instances nextLayerData){
        if(data.numInstances()==nextLayerData.numInstances() || nextLayerData.numInstances()==0){
            return true;
        }else{
            return false;
        }
    }


    protected Instances rulesetForOneLayer(Instances data)
	throws Exception {

        Instances newData = data, growData=new Instances(data);
        growData.delete();
		FastVector ruleset = new FastVector();

		aprioriDistribution = new double[newData.classAttribute().numValues()];// the instances number of each class
		boolean allWeightsAreOne = true;
		for (int i = 0 ; i < newData.numInstances(); i++){
			aprioriDistribution[(int)newData.instance(i).classValue()]+=newData.instance(i).weight();
			if (allWeightsAreOne && newData.instance(i).weight() != 1.0){
				allWeightsAreOne = false;
				break;
			}
		}

        if(m_Debug)
			System.err.println("\n*** Building stage ***");
        
        oneClass:
        for(int y=0; y < data.numClasses(); y++){ // For each class	
            double classIndex = (double)y;
            if(m_Debug){
                int ci = (int)classIndex;
                System.err.println("\n\nClass "+m_Class.value(ci)+"("+ci+"): "
                        + aprioriDistribution[y] + "instances\n"+
                "=====================================\n");
            }

            if(Utils.eq(aprioriDistribution[y],0.0)) // No data for this class
                continue oneClass;
            RipperRule oneRule;
            oneRule = new RipperRule(this.aprioriDistribution);
            oneRule.setConsequent(classIndex);  // Must set first
			oneRule.setAlpha(m_alpha);
			oneRule.setMaxAttUsed(m_maxAttUsed);
			oneRule.setShreshlod(m_shreshold);
            if(m_Debug)
                System.err.println("\ngrowing a rule ...");
            oneRule.grow(newData);             // Build the rule
            if(m_Debug)
                System.err.println("one rule found:\n"+
                        oneRule.toString(m_Class));
            ruleset.addElement(oneRule);
        }
		m_Ruleset.addElement(ruleset);
        for (int i=0; i<newData.numInstances();i++){
            Instance ins=(Instance)newData.instance(i);
            Boolean inThisLayer=false;
            for (int j=0;j<ruleset.size();j++){
                RipperRule rule = (RipperRule)ruleset.elementAt(j);
                if (rule.covers(ins)){
                    inThisLayer=true;
                    break;
                }
            }
            if (!inThisLayer){
                growData.add(ins);
            }
        }
        return growData;
    }

    	/**
	 * Classify the test instance with the rule learner and provide
	 * the class distributions 
	 *
	 * @param datum the instance to be classified
	 * @return the distribution
	 * @throws Exception 
	 */

	public double[] distributionForInstance(Instance datum) throws Exception{ 
		//test for multiple overlap of rules
		double[] rulesCoveringForEachClass = new double[datum.numClasses()];  
		for(int i=0; i < m_Ruleset.size(); i++){
			FastVector layerRules = (FastVector)m_Ruleset.elementAt(i);
            for (int j=0; j<layerRules.size();j++){
                RipperRule rule = (RipperRule) layerRules.elementAt(j);
                if (!rule.hasAntds()) 
				    continue;
                if(rule.covers(datum)||i==m_Ruleset.size()-1){
                    rulesCoveringForEachClass[(int)rule.m_Consequent] = rule.computeAverageMembershipDegree(datum);
                    //System.err.println("Cov: "+rule.getConfidence());
                }
			}
            if (Utils.sum(rulesCoveringForEachClass)!=0){
                break;
            }
                
    
		}


		//check for conflicts
		// double[] maxClasses = new double[rulesCoveringForEachClass.length];
		// for (int i = 0; i < rulesCoveringForEachClass.length; i++){
		// 	if (rulesCoveringForEachClass[Utils.maxIndex(rulesCoveringForEachClass)] ==
		// 		rulesCoveringForEachClass[i] && rulesCoveringForEachClass[i]>0)
		// 		maxClasses[i] = 1;
		// }

		// if (Utils.sum(maxClasses)>0){
		// 	for (int i = 0; i < maxClasses.length; i++){
		// 		maxClasses[i] *= aprioriDistribution[i]/Utils.sum(aprioriDistribution);
		// 	}
		// 	rulesCoveringForEachClass=maxClasses;
		// }


		// // If no stretched rule was able to cover the instance,
		// // then fall back to the apriori distribution
		// if (Utils.sum(rulesCoveringForEachClass)==0){
		// 	rulesCoveringForEachClass = aprioriDistribution;
		// }

		// if (Utils.sum(rulesCoveringForEachClass)>0)
		// 	Utils.normalize(rulesCoveringForEachClass);

		return rulesCoveringForEachClass;

	}

    /**
	 * It writes the training and test files with the real and predicted classes
	 * @param outputVal String The string with the training (validation) output
	 * @param outputTest String The string with the test output
	 * @param accTrain double The accuracy rate in training
	 * @param accTest double The accuracy rate in test
	 */
	void writeOutput(String outputVal, String outputTest, double accTrain, double accTest){
		String p = new String("");
		p = "@relation " + Attributes.getRelationName() + "\n";
		p += Attributes.getInputAttributesHeader();
		p += Attributes.getOutputAttributesHeader();
		p += Attributes.getInputHeader() + "\n";
		p += Attributes.getOutputHeader() + "\n";
		p += "@data\n";
		Files.writeFile(outputTrainFile,p+outputVal);
		Files.writeFile(outputTestFile,p+outputTest);
		System.out.println("Training accuracy: "+accTrain);
		System.out.println("Test accuracy: "+accTest);
		System.out.println("m_Optimizations = " + m_Optimizations);
	}

    	/**
	 * Returns the value of the named measure
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) 
			return m_Ruleset.size();
		else 
			throw new IllegalArgumentException(additionalMeasureName+" not supported (RIPPER)");
	}  

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String foldsTipText() {
		return "Determines the amount of data used for pruning. One fold is used for "
		+ "pruning, the rest for growing the rules.";
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String optimizationsTipText() {
		return "The number of optimization runs.";
	}

	/**
	 * Sets the number of optimization runs
	 * 
	 * @param run the number of optimization runs
	 */
	public void setOptimizations(int run) {
		m_Optimizations = run;
	}

	/**
	 * Gets the the number of optimization runs
	 * 
	 * @return the number of optimization runs
	 */
	public int getOptimizations() {
		return m_Optimizations;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String debugTipText() {
		return "Whether debug information is output to the console.";
	}

	/**
	 * Sets whether debug information is output to the console
	 * 
	 * @param d whether debug information is output to the console
	 */
	public void setDebug(boolean d) {
		m_Debug = d;
	}

	/**
	 * Gets whether debug information is output to the console
	 * 
	 * @return whether debug information is output to the console
	 */
	public boolean getDebug(){
		return m_Debug;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String checkErrorRateTipText() {
		return "Whether check for error rate >= 1/2 is included" +
		" in stopping criterion.";
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String useRuleStretchingTipText() {
		return "Whether rule stretching is performed.";
	}


	/**
	 * Prints the all the rules of the rule learner.
	 *
	 * @return a textual description of the classifier
	 */
	public String toString() {
		if (m_Ruleset == null) 
			return "FRBDT: No model built yet.";

		StringBuffer sb = new StringBuffer("FRBDT rules:\n"+
		"===========\n\n"); 
		for(int j=0; j<m_Ruleset.size(); j++){
			FastVector layerRules = (FastVector)m_Ruleset.elementAt(j);
			sb.append("The "+j+"-th layer rules:");
			for(int k=0; k<layerRules.size(); k++){
				//System.out.println("rules size = " + rules.size());
				sb.append(((RipperRule)layerRules.elementAt(k)).toString(m_Class)
						 +"\n");
			}			    
		}
		sb.append("\nNumber of layers : " 
				+ m_Ruleset.size() + "\n");

		return sb.toString();
	}
	/** 
	 * Get the ruleset generated by Ripper 
	 *
	 * @return the ruleset
	 */
	public FastVector getRuleset(){ return m_Ruleset; }


    public void setMaxAttUsed(int maxl){
        m_maxAttUsed=maxl;
      }
    
      public int getMaxAttUsed(){ return m_maxAttUsed;}
    
      public void setAlpha(double alpha){
        m_alpha=alpha;
      }
    
      public double getAlpha(){return m_alpha;}
    
      public void setShreshlod(double shreshold){
        m_shreshold=shreshold;
      }
    
      public double getShreshold(){return m_shreshold;}

 }


