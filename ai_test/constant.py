class BaseConstant:
    API_ENDPOINT = "https://qc-ai-test-api.rainscales.xyz/api/testing/detect/"
    ACCESS_KEY = '6nVgb14aldZFzAcKWCpL'
    SECRET_KEY = 'iHDHK35yhHgnWADfXejsEzySUqKi7KXcczN0zMDn'
    BUCKET_NAME = 'ai-testing'
    S3_ENDPOINT = 'http://minio:9000'
    IOU_THRESHOLD = 0.5
    AI_CONFIG = {0: "Forklift",
                 1: "Hand pallet jack",
                 2: "Electric pallet jack",
                 3: "Reach truck",
                 4: "Truck",
                 5: "Pallet",
                 6: "Product box",
                 7: "Product package",
                 8: "Fallen package",
                 9: "Person",
                 10: "Person wear visible clothes",
                 11: "Person using phone",
                 12: "Person eating or drinking",
                 13: "Person carrying object",
                 14: "Person pull object",
                 15: "Alcohol testing tool",
                 16: "Firefighting equipment",
                 17: "Wheel Chocks",
                 18: "Beacon light",
                 19: "No_beacon light",
                 20: "Security person"}
    DATA_LABEL_MAPPING = {
        1: 0,  # forklift -> Forklift
        2: 1,  # hand_pallet_jack -> Hand pallet jack
        3: 2,  # electric_pallet_jack -> Electric pallet jack
        4: 3,  # reach_truck -> Reach truck
        5: 7,  # product_package -> Product package
        6: 5,  # pallet -> Pallet
        12: 15,  # Alcohol_testing_tool -> Alcohol testing tool
        13: 4,  # truck -> Truck
        14: 9,  # person -> Person
        15: 18  # Beacon_light -> Beacon light
    }

cfg = BaseConstant()
