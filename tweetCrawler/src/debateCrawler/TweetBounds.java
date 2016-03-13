package debateCrawler;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

// TweetBounds - save off current tweet query boundaries
//               and restore for next run.
public class TweetBounds
{

    public long sinceID;
    public long maxID;

    // the boundary is query specific
    protected int queryId;
    protected String query;

    
    // The prefix lets us identify the file easier
    protected String prefix;

    protected TweetBounds()
    {
        this.sinceID = 0;
        this.maxID = -1;
        this.query = "";
        this.queryId = 0;
        this.prefix = "";

    }

    public TweetBounds(String query)
    {
        this.sinceID = 0;
        this.maxID = -1;
        this.query = query;
        this.queryId = query.hashCode();
        this.prefix = "";

    }

    public TweetBounds(String query, String prefix)
    {
        this.sinceID = 0;
        this.maxID = -1;
        this.query = query;
        this.queryId = query.hashCode();
        if (prefix == null || prefix.isEmpty())
        {
            this.prefix = "";
        } else
        {
            this.prefix = prefix + "_";
        }

    }

    public TweetBounds(long since, long max, String query, String prefix)
    {
        this.sinceID = since;
        this.maxID = max;
        this.query = query;
        this.queryId = query.hashCode();


        if (prefix == null || prefix.isEmpty())
        {
            this.prefix = "";
        } else
        {
            this.prefix = prefix + "_";
        }
    }

    protected String getFileName()
    {
        return this.prefix + "CachedBounds_" + this.queryId + ".txt";
    }

    public void initTweetBounds() throws IOException
    {
        String filename = getFileName();
        File cacheFile = new File(filename);

        if (!cacheFile.exists())
        {
            // Reset bounds to defaults.
            this.sinceID = 0;
            this.maxID = -1;

            return;
        }

        BufferedReader br = null;
        try
        {
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
        catch (FileNotFoundException ex)
        {
            System.err.println("File not found: " + filename);
            throw ex;
        }
        finally
        {
            if (br != null)
            {
                br.close();
            }
        }

        return;
    }

    public void saveTweetBounds() throws IOException
    {
        saveTweetBounds(null);
    }
    
    public void saveTweetBounds(String debugInfo) throws IOException
    {
        String filename = getFileName();

        BufferedWriter wr = null;
        try
        {
            wr = new BufferedWriter(new FileWriter(filename));

            String firstLine = this.sinceID + "," + this.maxID;

            // Note: this overwrites the first line in the file.
            wr.write(firstLine);

            // We write the query for debugging - not used on init.
            wr.newLine();
            wr.write(this.query);
            
            if (debugInfo != null && !debugInfo.isEmpty())
            {
                wr.newLine();
                wr.write(debugInfo);
            }

        }
        catch (FileNotFoundException ex)
        {
            System.err.println("File not found: " + filename);
            throw ex;
        }
        finally
        {
            if (wr != null)
            {
                wr.flush();
                wr.close();
            }
        }

        return;

    }
}
