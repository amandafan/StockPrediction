import sys

class ReadFile:

  def __init__(self, date):
    self.news = self.get_news(date);
    self.price = self.get_price(date);

  # Read Files and get all the news published on the given date
  def get_news(self, date):
    return "";

  # Read Files and get the closing price of the given date
  def get_price(self, date):
    return 0;
