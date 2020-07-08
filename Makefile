.PHONY : app clean clean_res

app:
	python oremoremo.py

clean:
	rm -rf *.log

clean_res:
	rm -rf result