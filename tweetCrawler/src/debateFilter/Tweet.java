package debateFilter;

import com.opencsv.bean.CsvBind;

public class Tweet {
	
	@CsvBind
	private String tweet_id;
	
	@CsvBind
	private String candidate;
	
	@CsvBind
	private String name;
	
	@CsvBind
	private String tweet_created;
	
	@CsvBind
	private String tweet_location;
	
	@CsvBind
	private String user_timezone;
	
	@CsvBind
	private String retweet_count;
	
	@CsvBind
	private String text;

	public String getTweet_id() {
		return tweet_id;
	}

	public String getCandidate() {
		return candidate;
	}

	public String getName() {
		return name;
	}

	public String getTweet_created() {
		return tweet_created;
	}

	public String getTweet_location() {
		return tweet_location;
	}

	public String getUser_timezone() {
		return user_timezone;
	}

	public String getRetweet_count() {
		return retweet_count;
	}

	public String getText() {
		return text;
	}
	
	
}
