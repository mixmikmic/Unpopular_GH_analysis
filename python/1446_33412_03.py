# !pip install --user ipython-sql
get_ipython().magic('load_ext sql')

# Connect to the Chinook database
get_ipython().magic('sql sqlite:///Chinook.sqlite')

get_ipython().run_cell_magic('sql', '', "\n-- Find the ArtistId for 'Deep Purple'\nSELECT *\nFROM Artist\nWHERE Name = 'Deep Purple';")

get_ipython().run_cell_magic('sql', '', 'SELECT *\nFROM Album\nWHERE ArtistId = 58;')

get_ipython().run_cell_magic('sql', '', '\nSELECT Name, Composer\nFROM Track\nWHERE AlbumId = 50;')

get_ipython().run_cell_magic('sql', '', "\nSELECT *\nFROM Artist, Album\nWHERE Artist.ArtistId = Album.ArtistId\nAND Artist.NAME = 'Deep Purple';")

get_ipython().run_cell_magic('sql', '', "\nSELECT Track.Name, Track.Composer \nFROM Artist, Album, Track\nWHERE Album.ArtistId = Artist.ArtistId\nAND Track.AlbumId = Album.AlbumId\nAND Artist.Name = 'Deep Purple';")

