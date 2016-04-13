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

	public void setTweet_id(String tweet_id) {
		this.tweet_id = tweet_id;
	}

	public void setCandidate(String candidate) {
		this.candidate = candidate;
	}

	public void setName(String name) {
		this.name = name;
	}

	public void setTweet_created(String tweet_created) {
		this.tweet_created = tweet_created;
	}

	public void setTweet_location(String tweet_location) {
		this.tweet_location = tweet_location;
	}

	public void setUser_timezone(String user_timezone) {
		this.user_timezone = user_timezone;
	}

	public void setRetweet_count(String retweet_count) {
		this.retweet_count = retweet_count;
	}

	public void setText(String text) {
		this.text = text;
	}
	
	
}
