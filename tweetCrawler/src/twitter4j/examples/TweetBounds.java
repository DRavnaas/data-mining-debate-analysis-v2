package twitter4j.examples;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class TweetBounds {
    
    public long sinceID;
    public long maxID;
    
    // the boundary is query specific
    protected String query;

    public TweetBounds()
    {
        sinceID = 0;
        maxID = -1;
    }
    
    public TweetBounds(long since, long max)
    {
        sinceID = since;
        maxID = max;
    }
    
    public void initTweetBounds(String query) throws IOException
    {
        this.sinceID = 0;
        this.maxID = -1;
        this.query = query;
        
        // Borrow some of Yogi's filebase code?
        String filename = "CachedBounds_" + query.hashCode() + ".txt";
        File cacheFile = new File(filename);
        
        if (!cacheFile.exists())
        {
            return;
        }
        
        BufferedReader br=null;
        try{
            br = new BufferedReader(new FileReader(filename));
            
            String firstLine = br.readLine();
            if (firstLine != null)
            {
                String[] values = firstLine.split(",");
                if (values.length > 1)
                {
                    this.sinceID = Long.parseLong(values[0]);
                    this.maxID = Long.parseLong(values[1]);
                }                
            }      
            
        }
        catch(FileNotFoundException ex){
            System.err.println("File not found: "+filename);
            throw ex;
        }
        finally {
            if (br != null)
            {
                br.close();
            }
        }
        
        return;        
    }
    
    public void saveTweetBounds() throws IOException
    {
        // Borrow some of Yogi's filebase code?
        String filename = "CachedBounds_" + query.hashCode() + ".txt";
        
        BufferedWriter wr=null;
        try{
            wr = new BufferedWriter(new FileWriter(filename));

            String firstLine = this.sinceID + "," + this.maxID;

            // Note: this overwrites the first line in the file.
            wr.write(firstLine);
            
            // We write the query for debugging - not used on init.
            wr.newLine();
            wr.write(this.query);
            
        }
        catch(FileNotFoundException ex){
            System.err.println("File not found: "+filename);
            throw ex;
        }
        finally {
            if (wr != null)
            {
                wr.flush();
                wr.close();
            }
        }
        
        return;             
        
    }
}
