package ConvertToCSV;


import twitter4j.*;
import twitter4j.Query.ResultType;
import twitter4j.conf.ConfigurationBuilder;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Properties;

import com.opencsv.*;

public final class ConvertToCSV {
    public static void main(String[] args) {
    	if (args.length != 1) {
    		System.out.println("Usage: ConvertToCSV JSON_source_dir");
    		System.exit(-1);
    	}
    	try {
    		File[] files = new File(args[0]).listFiles(new FilenameFilter() {
                 //@Override
                 public boolean accept(File dir, String name) {
                     return name.endsWith(".json");
                 }
             });
             for (File file : files) {
            	 List<Status> restoredTweets = readLines(file);
            	 saveToCSV(restoredTweets, file.getName().replace(".json", ".csv"), false);
                 System.out.println("saved " + file.getName() +" to " + file.getName().replace(".json", ".csv"));
             }
             System.exit(0);
         } catch (Exception e) {
             e.printStackTrace();
             System.out.println("Failed to store tweets: " + e.getMessage());
             System.exit(-1);
         }
 
 }

    public static void saveToCSV(List<Status> tweets, String targetCSV, boolean filterOutRetweet) throws Exception{
    	CSVWriter writer = null;
    	CSVReader reader = null;
    	try {
    		new File("savedTweets").mkdir();
    		

    		if (targetCSV.endsWith(".csv") || targetCSV.endsWith(".CSV"))
    			targetCSV = targetCSV.substring(0, targetCSV.length()-4);
    		File targetCSVFile = new File("savedTweets/"+targetCSV+".csv");
    		

    		Hashtable<String, String[]> lookupTable = new Hashtable<String, String[]>();
    		

    		if (targetCSVFile.exists()) {
    			reader = new CSVReader(new FileReader(targetCSVFile));
    		
    			String [] nextLine;
	    	    while ((nextLine = reader.readNext()) != null) {
	    	        // nextLine[] is an array of values from the line
	    	    	lookupTable.put(nextLine[0], nextLine);
	    	    }
	    	    reader.close();
	    	    reader = null;
	    	    
	    	    writer = new CSVWriter(new FileWriter(targetCSVFile,true));
    		}
    		else {
    			writer = new CSVWriter(new FileWriter(targetCSVFile,true));
    			String[] headers = {"tweet_id", "name","tweet_created","tweet_location","user_timezone","text"};
    			writer.writeNext(headers);
    		}
    		
    		
    		
    		SimpleDateFormat formatter = new SimpleDateFormat("MM/dd/yyyy HH:mm"); 

            for (Status status : tweets) {
            	 //if retweet need to be filtered out.
            	 if ((filterOutRetweet)&& status.isRetweet())
            		 continue;
            	 //specify fields to save to CSV
				String[] tweetRecord={String.valueOf(status.getId()),
            			 				status.getUser().getName(), 
            			 				formatter.format(status.getCreatedAt()), 
            			 				status.getUser().getLocation(),
            			 				status.getUser().getTimeZone(),
            			 				status.getText()};
            	 
            	 if (lookupTable.get(tweetRecord[0])== null) {
            		 lookupTable.put(tweetRecord[0], tweetRecord);
            		 writer.writeNext(tweetRecord);
            	 }
            }
         } finally {
        	 if (reader != null) {
                 try {
                	 reader.close();
                 } catch (IOException ignore) {
                 }
              }
        	 
             if (writer != null) {
                 try {
                	 writer.flush();
                	 writer.close();
                 } catch (IOException ignore) {
                 }
              }
          }
            
    }
    
    private static List<Status> readLines(File fileName) throws Exception {
        FileInputStream fis = null;
        InputStreamReader isr = null;
        BufferedReader br = null;
        List<Status> ret = new ArrayList<Status>();
        try {
            fis = new FileInputStream(fileName);
            isr = new InputStreamReader(fis, "UTF-8");
            br = new BufferedReader(isr);
            br.readLine();
            String nextLine = br.readLine();
            while(nextLine!=null) {
            	try {
					ret.add(TwitterObjectFactory.createStatus(nextLine));
				} catch (TwitterException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            	nextLine = br.readLine();
            }
         return ret;
            
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException ignore) {
                }
            }
            if (isr != null) {
                try {
                    isr.close();
                } catch (IOException ignore) {
                }
            }
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException ignore) {
                }
            }
        }
    }

}