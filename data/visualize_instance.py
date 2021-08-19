import matplotlib.pyplot as plt
import numpy as np
import argparse
import pkg_resources

# parse in a full sample vector and locations

def load_vector_with_locations(vector, locations, covg):
    vect, locs = np.load(vector, allow_pickle=True, encoding="bytes"), np.load(locations).astype(str)
    bag = []
    
    #preprocess and normalize vector
    for tumor_normal_microsatellite in vect:
        tumor = tumor_normal_microsatellite[0]
        normal = tumor_normal_microsatellite[1]

        # downsample to the required coverage
        downsampled_t = tumor[
            np.random.choice(tumor.shape[0],covg), :, :
        ]
        downsampled_n = normal[
            np.random.choice(normal.shape[0], covg), :, :
        ]

        # stack tumor and normal 
        bag.append(np.concatenate((downsampled_t, downsampled_n), axis=0))
    
    # concat to consistent dimensions and normalize
    bag = [np.concatenate((entry, np.zeros((200, 40 - entry.shape[1], 3))),axis=1,) for entry in bag]
    bag = np.array(bag)
    bag_mean = bag.mean(axis=(0, 1, 2), keepdims=True)
    bag_std = bag.std(axis=(0, 1, 2), keepdims=True)

    # Normalize bag
    bag = (bag - bag_mean) / (bag_std)
    return bag, locs


# get one particular instance

def get_instance_by_location(vector, locations_list, coordinates):
    vect_index = locations_list.index(coordinates)
    instance = vector[vect_index]
    return instance

# visualize instances

def visualize_instances(sites_to_view, vector, locations, figname):

    f, axarr = plt.subplots(len(sites_to_view), 3, figsize=(12,64))
    f.tight_layout()

    for j in range(len(sites_to_view)):
        instance = get_instance_by_location(vector, locations, sites_to_view[j])

        for i in range(instance.shape[2]):
            if len(sites_to_view) > 1:
                axarr[j][i].set_title(str(sites_to_view[j]) + " channel " + str(i))
                axarr[j][i].imshow(instance[:,:,i], aspect='auto')
            else:
                axarr[i].set_title(str(sites_to_view[j]) + " channel " + str(i))
                axarr[i].imshow(instance[:,:,i], aspect='auto')

    f.savefig(figname + ".pdf")
    return axarr




def main():
    parser = argparse.ArgumentParser(description="MiMSI Site Visualization Utility")
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="Display current version of MiMSI",
    )
    parser.add_argument(
        "--vector",
        default="./test-data.npy",
        help="Vector .npy for the case you'd like to visualize",
    )
    parser.add_argument(
        "--locations",
        default="./test-locations.npy",
        help="Locations .npy for the case you'd like to visualize",
    )
    parser.add_argument(
        "--site",
        help="Site to visualize, must be present in locations file for the image to generate properly",
    )
    parser.add_argument(
        "--site-list",
        help="File indicating the site(s) to visualize ",
    )
    parser.add_argument(
        "--coverage",
        default=100,
        help="Required coverage for both the tumor and the normal. Any coverage in excess of this limit will be randomly downsampled",
    )
    parser.add_argument(
        "--output",
        help="Name of the output filename",
    )
    

    args = parser.parse_args()

    if args.version:
        print("MiMSI Site Visualization Utility version - " + pkg_resources.require("MiMSI")[0].version)
        return 

    vector, locations, site, site_list, coverage, output_filename = (
        args.vector,
        args.locations,
        args.site,
        args.site_list,
        args.coverage,
        args.output
    )

    numpy_vector, numpy_locations = load_vector_with_locations(vector, locations, int(coverage))

    locations_to_generate = []
    list_locations = []

    for loci in numpy_locations:
        list_locations.append(loci[0] + "," + loci[1] + "," + loci[2])
    
    if site_list is not None:
        with open(site_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                locations_to_generate.append(line.strip())
    else:
        locations_to_generate.append(site)

    viz = visualize_instances(locations_to_generate, numpy_vector, list_locations, output_filename)
    # todo save viz


if __name__ == "__main__":
    main()