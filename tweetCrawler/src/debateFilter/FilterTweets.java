package debateFilter;
// This program reads all the file in the input directory, filter out the tweets if
//   - the candidates under which tweet is categorized is not present in the tweet  OR
//   - Multiple candidates are mentioned in the tweets
// and then, writes out the filtered tweets

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.bean.BeanToCsv;
import com.opencsv.bean.ColumnPositionMappingStrategy;
import com.opencsv.bean.CsvToBean;
import com.opencsv.bean.HeaderColumnNameMappingStrategy;

public class FilterTweets {
	
	//SPECIFY THE INPUT FILE DIRECTORY HERE....
	static String dataDirectoryPath = "data/march";
	// SPECIFY THE OUTPUT FILE DIRECTORY HERE....
	static String filteredDirectoryPath = "filtered/march";

	static int BATCH_SIZE = 1000;
	// Maps to save the regex 
	static Map<String,String> regexMultCandFilter = new HashMap<String,String>();
	static Map<String,String> regexCurCandFilter = new HashMap<String,String>();
	
	public static void main(String[] args) throws IOException {		
		// List of candidates in the CSV to be filtered. No need to change if any of the candidate does not appear in the file.
		List<String> candidates = new ArrayList<String>(Arrays.asList("trump","cruz","rubio","kasich","carson"));
		
		// Build the Regex for filtering 
		buildRegex(candidates);
		
		// Process each file in the data directory
		File dir = new File(dataDirectoryPath);
		  File[] directoryListing = dir.listFiles();
		  if (directoryListing != null) {
			   
			  for (File file : directoryListing) {
			    	String filename = file.getPath();
			    	
			    	// Construct the filtered file name
			    	String filteredFileName = file.getName();
			    	filteredFileName = filteredFileName.substring(0, filteredFileName.length()-4);
			    	filteredFileName =  filteredDirectoryPath + "/" + filteredFileName + "_filtered.csv";
			    	
			    	filterFile(filename,filteredFileName);
			    }
		  }
		  else{
		    System.out.println("Empty directory " + dataDirectoryPath );
		  }
	
	}
	
	public static void filterFile(String filename,String filteredFileName) throws IOException {		

		// Read the CSV and form the tweet list
		List<Tweet> tweetList = readCSV(filename);
		
		System.out.println("Processing " + filename + "....." );
		System.out.println("TOTAL NO. OF RECORDS : " + tweetList.size());	

		// A list to store the filtred tweets to write the file in batch
		List<Tweet> filteredTweetList = new ArrayList<Tweet>();

		int countFiltered = 0;		
		
		// Create a new file to store the result
		initCSVFile(filteredFileName);  
		// For each tweet, write it to the new file if it passes filtering criteria
		
		for(Tweet tweet: tweetList){
			// DON'T write to file if candidate name is not mentioned 
			// OR if multiple candidates are mentioned
			try{			
			if(candidateNameNotMentioned(tweet.getCandidate(),tweet.getText()) || 
					multipleCandidatesMentioned(tweet.getCandidate(),tweet.getText())){
				//System.out.println(tweet.getCandidate() + "****" + tweet.getText());
				countFiltered++;
			}
			else{
				filteredTweetList.add(tweet);
				// Write to the CSV in batch
				if(filteredTweetList.size() == BATCH_SIZE){
					writetoCSV(filteredTweetList,filteredFileName,true);
					filteredTweetList.clear();
				}
			}
			}
			catch(Exception e){
				System.out.println("Error in processing tweet: " + tweet.getName() +" candidate: " + tweet.getCandidate() + "  text: " + tweet.getText() );
				e.printStackTrace();
			}
		}
		
		// If there are any remaining tweets to be written, write now
		if(! filteredTweetList.isEmpty()){
			writetoCSV(filteredTweetList,filteredFileName,true);
			filteredTweetList.clear();
		}
		
		System.out.println("NO. OF RECORDS FILTERED: " + countFiltered);	
	}

/* Helper Methods */
	
	// Filter 1: Returns true if input candidate name is not mentioned
	private static boolean candidateNameNotMentioned(String candidate, String text) {
		Pattern p ;
		Matcher m=null;
		
		if(candidate != null){
			 p = Pattern.compile(regexCurCandFilter.get(candidate.toLowerCase()));
			 m = p.matcher(text);
		}
		return ! m.find();
	}

		
	// Filter 2: Returns true if multiple candidates are mentioned i.e. any other candidate apert from
	// input candidate is passed 	
	private static boolean multipleCandidatesMentioned(String candidate, String text) {
		candidate.toLowerCase();
		Pattern p = Pattern.compile(regexMultCandFilter.get(candidate.toLowerCase()));
		Matcher m = p.matcher(text);
		
		return m.find();
	}

	
	// Reads the CSV, maps each row the 'Tweet'	 bean and then returns the list of tweets
	private static List<Tweet> readCSV(String filename) throws FileNotFoundException {
		CSVReader reader = new CSVReader(new FileReader(filename));

		HeaderColumnNameMappingStrategy<Tweet> strategy = new HeaderColumnNameMappingStrategy<Tweet>();
		strategy.setType(Tweet.class);
		CsvToBean<Tweet> csvToBean = new CsvToBean<Tweet>();
		List<Tweet> tweetList = csvToBean.parse(strategy,reader);
		return tweetList;		
	}
	
//	private static List<Tweet> readCSV(String filename) throws IOException {
//	     CSVReader reader = new CSVReader(new FileReader(filename));
//	     String [] nextline;
//	     List<Tweet> tweetList = new ArrayList<Tweet>();
//	     
//	     while( (nextline = reader.readNext()) != null){
//	    	 Tweet tweet = new Tweet();    	 
//	    	 try {
//	    		 tweet.setTweet_id(nextline[0]);
//		    	 tweet.setCandidate(nextline[1]);
//		    	 tweet.setName(nextline[2]);
//		    	 tweet.setTweet_created(nextline[3]);
//		    	 tweet.setTweet_location(nextline[4]);
//		    	 tweet.setUser_timezone(nextline[5]);
//		    	 tweet.setRetweet_count(nextline[6]);
//		    	 tweet.setText(nextline[7]);
//	    	 }
//	    	 catch(Exception e) {
//	    		 System.out.println("Exception : " + nextline[0]);
//	    		 e.printStackTrace();
//	    	 }
//	    	
//	    	 
//	    	 tweetList.add(tweet);
//	    	 
//	    }
//	     reader.close();
//	return tweetList;		
//}

	
	// Initializes the output file
	private static void initCSVFile(String filename) throws IOException {		   		
	      CSVWriter writer = new CSVWriter(new FileWriter(filename,false));
	      writer.close();
	}
	
	// Writes the input list of filtered tweets to the file name provided
	private static void writetoCSV(List<Tweet> filteredTweetList,
		  String filename, boolean appendFile) throws IOException {		
	   		
	      CSVWriter writer = new CSVWriter(new FileWriter(filename,appendFile));

	      ColumnPositionMappingStrategy<Tweet> strat = new ColumnPositionMappingStrategy<Tweet>();
	      strat.setType(Tweet.class);
	      String[] columns = new String[]{"tweet_id", "candidate", "name","tweet_created","tweet_location","user_timezone","retweet_count","text"};
	      strat.setColumnMapping(columns);
	      
	      BeanToCsv<Tweet> annotatedBean= new BeanToCsv<Tweet>();
	      annotatedBean.write(strat, writer, filteredTweetList);     
	      
		  writer.close();
		      
	}
	
	// This method build the regex required for filtering based on the candidate list provided
	private static void buildRegex(List<String> candidates) {
		
		// Example for Cruz : 		
		// regexCurCandFilter = "(?i)^?(cruz)";
		// regexMultCand = "(?i)^?(rubio|trump|carson|kasich)";
		
		String regexMultCand;
		String regexCurcand;
 		
		for (String curCand :candidates ){
			 regexMultCand = "(?i)^?(";
			
			 for(String candidate: candidates){
				if (!candidate.equalsIgnoreCase(curCand))
					regexMultCand = regexMultCand + candidate + "|";
			}
			
			regexMultCand = regexMultCand.substring(0, regexMultCand.length()-1);
			regexMultCand= regexMultCand + ")";
			
			regexCurcand = "(?i)^?(" + curCand + ")";
 			
			regexMultCandFilter.put(curCand, regexMultCand);
			regexCurCandFilter.put(curCand,regexCurcand);
			
			//System.out.println(regexCurcand);
			//System.out.println(regexMultCand);
		}	
	}
}
