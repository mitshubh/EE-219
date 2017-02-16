import pandas as pd
import matplotlib.pyplot as pyplot

def add_day_column(week_no, day):
	if (day == "Monday"):
		return (week_no-1)*7+1
	if (day == "Tuesday"):
		return (week_no-1)*7+2
	if (day == "Wednesday"):
		return (week_no-1)*7+3
	if (day == "Thursday"):
		return (week_no-1)*7+4
	if (day == "Friday"):
		return (week_no-1)*7+5
	if (day == "Saturday"):
		return (week_no-1)*7+6
	if (day == "Sunday"):
		return (week_no-1)*7+7


nw_data = pd.read_csv("./network_backup_dataset.csv")
#Extract a 20 day summary 
nw_data = nw_data.where((nw_data["Week #"]<3) | ((nw_data["Week #"]==3) & (nw_data["Day of Week"]!="Sunday"))).dropna()

#Extract subsequent 20 day summary
#nw_data = nw_data.where(((nw_data["Week #"]>3) & (nw_data["Week #"]<6)) | ((nw_data["Week #"]==6) & (nw_data["Day of Week"]!="Sunday"))).dropna()

#Create a column w.r.t. days
nw_data["Day #"] = nw_data.apply(lambda x: add_day_column(x["Week #"], x["Day of Week"]), axis=1)

#Extract relevant content out of the data
nw_data = nw_data[["Day #", "Work-Flow-ID", "Size of Backup (GB)"]]

fig, ax = pyplot.subplots()
for group, frame in nw_data.groupby(["Work-Flow-ID"]):
	nw_agg = frame.groupby(["Day #"]).sum()
	nw_agg = nw_agg.reset_index()
	#print (nw_agg)
	ax = nw_agg.plot(x="Day #", y="Size of Backup (GB)",  kind='line', ax=ax, label=group)
	
pyplot.legend()
pyplot.ylabel('Size of Backup (GB)')
ax.set_title('Size (GB) vs Day # - First 20 days')
pyplot.show()