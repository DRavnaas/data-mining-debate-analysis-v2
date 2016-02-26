package twitter4j.examples;

import twitter4j.*;

import java.io.*;
import java.util.List;

public class SaveTweets {
	//maxCount for each file, but not implemented strict split based on this count
	//split will only happen after whole tweets of a query are stored 
	//and total count exceeds maxCount. 
	private int maxCount;
	//fileBase should be a folderpath + filenameprefix 
	private String fileBase;
	//current file which is being written into.
	private String currentFile;
	//counter for when to store a new file.
	private int nextThreshold;
	
	public int getMaxCount() {
		return maxCount;
	}

	public void setMaxCount(int maxCount) {
		this.maxCount = maxCount;
	}

	public String getFileBase() {
		return fileBase;
	}

	public void setFileBase(String fileBase) {
		this.fileBase = fileBase;
	}

	public String getCurrentFile() {
		return currentFile;
	}

	public void setCurrentFile(String currentFile) {
		this.currentFile = currentFile;
	}
	//contruct function:
	//inputs: 
	//   maxCount: ideal maxCount for each singe file
	//   fileBase: folderpath + file_prefix
	SaveTweets(int maxCount, String fileBase) {
		this.maxCount = maxCount;
		this.fileBase = fileBase;
        this.currentFile = this.fileBase + "_0.json";
        this.nextThreshold = maxCount;
	}
	
	//Storing function
	//inputs:
	//    currentCount: current total count of tweets save for this filebase
	//    Result: query result to be store at this time. 
	public void storeQueryResult(int currentCount, QueryResult Result) {
		FileOutputStream fos = null;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;
        try {
        	//open file with append file set to be true
            fos = new FileOutputStream(currentFile,true);
            osw = new OutputStreamWriter(fos, "UTF-8");
            bw = new BufferedWriter(osw);
            for (Status status : Result.getTweets()) {
            	String rawJSON = TwitterObjectFactory.getRawJSON(status);
            	bw.write(rawJSON);
                bw.newLine();
                bw.flush();
           }    
        }catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to store tweets: " + e.getMessage());
        } finally {
        	//decide if a new file is needed by checking if the total count of 
        	//tweets exceeds the threshold to create a new file.
        	//if so set currentFile to next filename and increase threshold by maxcount
        	if (Result.getCount() + currentCount >= nextThreshold) {
        		nextThreshold += maxCount;
        		this.currentFile = this.fileBase + "_" + String.valueOf(currentCount+Result.getCount())+".json";
        	}
            if (bw != null) {
                try {
                    bw.close();
                } catch (IOException ignore) {
                }
            }
            if (osw != null) {
                try {
                    osw.close();
                } catch (IOException ignore) {
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException ignore) {
                }
            }
        }
	}  
}