from pathlib import Path
import pandas as pd
import json

# Point to where the GTFS archive data is stored.
GTFS_ARCHIVE_PARENT_DIR = Path().home() / "Documents" / "Hack Oregon" / "GTFS_archive_data"

# Loop over each archived dir. This takes a while.
print("****** BEGIN ******")
for ARCHIVE_DIR in GTFS_ARCHIVE_PARENT_DIR.iterdir():
    # Ignore any hidden dirs.
    if ARCHIVE_DIR.name.startswith('.'):
        continue
    else:
        print(f"Folder: {ARCHIVE_DIR.name}")

    # Load in the files that we want.
    # calendar_dates.txt
    try:
        file = 'calendar_dates.txt'
        dates_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # stop_times.txt
    try:
        file = 'stop_times.txt'
        times_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # trips.txt
    try:
        file = 'trips.txt'
        trips_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # Init the dict to store all the stop info.
    stops_by_time = {}
    count = 0
    # Look at each date - service_id combo.
    for name, group in dates_df.groupby(['date', 'service_id']):
        # Skip non S, U, W service ids.
        if name[1] not in ['S', 'U', 'W']:
            continue
        else:
            print(f"\tDate: {name[0]}\t Service ID: {name[1]}")

        # Find the trips and routes associated with that service on that date.
        trips = trips_df['trip_id'][trips_df['service_id'] == name[1]]

        # Look at each trip (i = index, r = row in the trips for this service id)
        for i, r in trips_df[['route_id', 'trip_id']][trips_df['service_id'] == name[1]].iterrows():
            # df of the stops associated with this trip
            stops = times_df[times_df['trip_id'] == r['trip_id']]

            # Look at each stop in the trip to assemble a dict of the stop times (as strings).
            for ind, row in stops.iterrows():
                # If that stop_id exists as a key in the dict.
                if stops_by_time.get(str(row['stop_id']), False):
                    # If that route exists as a key for the stop.
                    if stops_by_time[str(row['stop_id'])].get(str(r['route_id']), False):
                        # If that date exists as a key for the stop.
                        if stops_by_time[str(row['stop_id'])][str(r['route_id'])].get(str(name[0]), False):
                            # Add the stop time.
                            stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
                        else:
                            # Init the date as a list and add the stop time.
                            stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])] = []
                            stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
                    else:
                        # Init that route as a dict, init the date as a list, and add the stop time.
                        stops_by_time[str(row['stop_id'])][str(r['route_id'])] = {}
                        stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])] = []
                        stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
                # Else init that stop as a dict, init the route as a dict, init the date as a list, and add the stop time.
                else:
                    stops_by_time[str(row['stop_id'])] = {}
                    stops_by_time[str(row['stop_id'])][str(r['route_id'])] = {}
                    stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])] = []
                    stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
        count +=1
        if count >= 1:
            break

    # Write to a json for further analysis.
    EXPORT_PATH = ARCHIVE_DIR / f'{ARCHIVE_DIR.name}.json'
    print(f'\t\tEXPORT: {EXPORT_PATH.name}')
    with open(EXPORT_PATH, 'w') as fobj:
        json.dump(stops_by_time, fobj, indent=4)
    break
print("****** COMPLETE ******")

# Loop over each archived dir. This takes a while.
print("****** BEGIN ******")
for ARCHIVE_DIR in GTFS_ARCHIVE_PARENT_DIR.iterdir():
    # Ignore any hidden dirs.
    if ARCHIVE_DIR.name.startswith('.'):
        continue
    else:
        print(f"Folder: {ARCHIVE_DIR.name}")

    # Load in the files that we want.
    # calendar_dates.txt
    try:
        file = 'calendar_dates.txt'
        dates_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # stop_times.txt
    try:
        file = 'stop_times.txt'
        times_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # trips.txt
    try:
        file = 'trips.txt'
        trips_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # Init the dict to store all the stop info.
    stops_by_time = {}
    count = 0
    # Look at each date - service_id combo.
    for name, group in dates_df.groupby(['date', 'service_id']):
        # Skip non S, U, W service ids.
        if name[1] not in ['S', 'U', 'W']:
            continue
        else:
            print(f"\tDate: {name[0]}\t Service ID: {name[1]}")
            date_serv_id = '-'.join([str(name[0]), str(name[1])])

        # Find the trips and routes associated with that service on that date.
        trips = trips_df['trip_id'][trips_df['service_id'] == name[1]]

        # Look at each trip (i = index, r = row in the trips for this service id)
        for i, r in trips_df[['route_id', 'trip_id']][trips_df['service_id'] == name[1]].iterrows():
            # df of the stops associated with this trip
            stops = times_df[times_df['trip_id'] == r['trip_id']]

            # Look at each stop in the trip to assemble a dict of the stop times (as strings).
            for ind, row in stops.iterrows():
                # If that route exists as a key in the dict.
                if stops_by_time.get(str(r['route_id']), False):
                    # If that stop exists as a key for the route.
                    if stops_by_time[str(r['route_id'])].get(str(row['stop_id']), False):
                        # If that date exists as a key for the stop.
                        if stops_by_time[str(r['route_id'])][str(row['stop_id'])].get(date_serv_id, False):
                            # Add the stop time.
                            stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
                        else:
                            # Init the date as a list and add the stop time.
                            stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id] = []
                            stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
                    else:
                        # Init that route as a dict, init the date as a list, and add the stop time.
                        stops_by_time[str(r['route_id'])][str(row['stop_id'])] = {}
                        stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id] = []
                        stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
                # Else init that stop as a dict, init the route as a dict, init the date as a list, and add the stop time.
                else:
                    stops_by_time[str(r['route_id'])] = {}
                    stops_by_time[str(r['route_id'])][str(row['stop_id'])] = {}
                    stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id] = []
                    stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
        count +=1
        if count >= 1:
            break

    # Write to a json for further analysis.
    EXPORT_PATH = ARCHIVE_DIR / f'{ARCHIVE_DIR.name}.json'
    print(f'\t\tEXPORT: {EXPORT_PATH.name}')
    with open(EXPORT_PATH, 'w') as fobj:
        json.dump(stops_by_time, fobj, indent=4)
    break
print("****** COMPLETE ******")



