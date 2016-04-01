package fmatrix;

import java.io.IOException;
import java.io.StringReader;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.*;

import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.*;

public class VectorFactory {
	List<String> freqWords;
	List<ColumnMapping> mapp;
	String inputCSV;
	String outputCSV;
	String mappingTXT;
	String freqWdTXT;
	VectorFactory(){
		this.freqWords= new ArrayList<String>();
		this.mapp = new ArrayList<ColumnMapping>();
	}
	
	public void readFreqWords(String nextLine){
		String[] words = nextLine.split(","); 
		for (int i =1;i<words.length;i++)
			this.freqWords.add(words[i]);
	}
	
	public void getColMapping(String nextLine){
		String outputColStr = nextLine.split(",")[0]; 
        String inputColStr = nextLine.split(",")[1];
        boolean isTweetTextField;
        if (nextLine.split(",").length>2)
            isTweetTextField = true;
        else
            isTweetTextField = false;
        ColumnMapping cm = new ColumnMapping(inputColStr,outputColStr,isTweetTextField);
        this.mapp.add(cm);
	}
	
	
	public ArrayList<String> parseTweetIntoWordsList(String tweetText) throws Exception {
		StandardAnalyzer analyzer = new StandardAnalyzer();
		TokenStream ts =  analyzer.tokenStream("contents", new StringReader(tweetText));
		ts.reset();
		ArrayList<String> analyzer_tokenizdWords = new ArrayList<String>();
	    while (ts.incrementToken()) {
	    	analyzer_tokenizdWords.add(ts.getAttribute(CharTermAttribute.class).toString());
	    }
	    return analyzer_tokenizdWords;
	 
		/*tweetText = tweetText.replaceAll("\\s", " ");
		Pattern pattern = Pattern.compile("([\\-\\+]{0,1}\\d[\\d\\.]+)"                               //numbers
										+ "|((\\w+\\.)+\\w+)"                                           //http address
										//+ "|(((http://)|(https://)){0,1}(\\w+\\.)+\\w+[/\\w\\_\\:]*)" //http address
										+ "|(\\w+\\_\\w+)"                                            //words with _
										+ "|(\\w[\\w\\']+)"                                           //words with '
										+ "|(\\w+)");                                                 //other words
		Matcher matcher = pattern.matcher(tweetText);
		ArrayList<String> tokenizdWords = new ArrayList<String>();
		while (matcher.find()) {
			tokenizdWords.add(matcher.group(0));
		}

		return tokenizdWords;*/

	}
	private int checkFeature(String feature, ArrayList<String> tweetWords) {
		for (int i =0;i<tweetWords.size();i++) {
			if (tweetWords.get(i).compareTo(feature)==0)
				return 1;
			/*if ((tweetWords.get(i).indexOf(feature)==0)) 
				if (feature.length()==1 && tweetWords.get(i).length()>1) {}
				else
					return 1;
			
			else if ((tweetWords.get(i).indexOf(feature)>0) && (feature.length()>1))
				return 1;*/
		}
		return 0;
	}
	private String[] formatCSVRecord(String[] unformattedRec) throws Exception {
		String[] formattedRec = new String[this.freqWords.size()+this.mapp.size()];
		String lowerCaseTweet = null;
		for (int i=0;i<this.mapp.size();i++) {
			if (mapp.get(i).srcColIdx!=-1)
				formattedRec[i] = unformattedRec[mapp.get(i).srcColIdx];
			if (mapp.get(i).isTweetTextField == true)
				lowerCaseTweet = unformattedRec[mapp.get(i).srcColIdx].toLowerCase();
		}

		ArrayList<String> tweetWords = this.parseTweetIntoWordsList(lowerCaseTweet);
		
		for (int i=0;i<this.freqWords.size();i++){
			formattedRec[i+this.mapp.size()] = String.valueOf(this.checkFeature(this.freqWords.get(i), tweetWords));
		}
		
		//for debug
		/*String test_list = "";
		for (int i=0;i<tweetWords.size();i++){
			test_list += tweetWords.get(i) + " ";
		}
		formattedRec[0] = test_list;*/
		
		
		return formattedRec;
	}
	private void mapColToIdx(String[] inputCsvHeader) {
		for (int i =0;i<inputCsvHeader.length;i++){
			for (int j=0;j<this.mapp.size();j++){
				if (this.mapp.get(j).srcCol.compareTo(inputCsvHeader[i])==0) {
					this.mapp.get(j).srcColIdx = i;
					break;
				}
			}
		}
	}

	public int parseTweetsToVector() throws Exception  {
		ArrayList<String[]> inputData = WrappedFileReader.readCsvFile(inputCSV);
		ArrayList<String[]> outputData = new ArrayList<String[]>();
        String[] outputHeader = new String[mapp.size()+this.freqWords.size()];
        for (int i = 0;i<mapp.size();i++) {
        	outputHeader[i] = mapp.get(i).tgtCol;}
        for (int i =0;i<this.freqWords.size();i++)
        	outputHeader[i+mapp.size()] = this.freqWords.get(i);
        outputData.add(outputHeader);
        this.mapColToIdx(inputData.get(0));
        for (int i =1;i<inputData.size();i++) {
        	String[] formattedData = this.formatCSVRecord(inputData.get(i));
        	outputData.add(formattedData);
        }
        WrappedFileReader.writeCsvFile(outputCSV,outputData);
        return 0;
	}
	
	public void initFileStrings(String line) {
		String key = line.split(",")[0];
		String value = line.split(",")[1];
		if (key.compareTo("input_csv")==0)
			this.inputCSV = value;
		if (key.compareTo("output_csv")==0)
			this.outputCSV = value;
		if (key.compareTo("mapping_file")==0)
			this.mappingTXT = value;
		if (key.compareTo("word_list")==0)
			this.freqWdTXT = value;
	}
	
	public static void main(String[] args) throws Exception{
		String dbg_args0= "config/config.txt";
		VectorFactory gv = new VectorFactory();
		Class[] parameterTypes = new Class[1];
        parameterTypes[0] = String.class;
        //init all the files needed by this program from a config file
        Method method1 = VectorFactory.class.getMethod("initFileStrings", parameterTypes);
        if (WrappedFileReader.readTxtFile(dbg_args0, method1, gv, false)==-1)
			System.exit(-1);
        
        //read frequent word list txt
        method1 = VectorFactory.class.getMethod("readFreqWords", parameterTypes);
		if (WrappedFileReader.readTxtFile(gv.freqWdTXT, method1, gv, true)==-1)
			System.exit(-1);
		
		//read column mapping relationship between input csv and output csv 
		method1 = VectorFactory.class.getMethod("getColMapping", parameterTypes);
		if (WrappedFileReader.readTxtFile(gv.mappingTXT, method1, gv, true)==-1)
			System.exit(-1);
		
		//parse input data 
		if (gv.parseTweetsToVector()==-1)
			System.exit(-1);
		System.exit(0);
	}
}


