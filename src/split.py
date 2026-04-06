# Inter-Patient Split Logic
# DS1: Training + Validation
# DS2: Testing (Never touched during training)

# Original DS1 Records (Mixed Train/Test in some papers, but here we treat as Development Set)
DS1_RECORDS = [
    '101','106','108','109','112','114','115','116',
    '118','119','122','124',
    '201','203','205','207','208','209',
    '215','220','223','230'
]

# DS2 Records (Strictly held out Test Set)
TEST_RECORDS = [
    '100','103','105','111','113','117',
    '121','123',
    '200','202','210','212','213','214',
    '219','221','222','228','231','232','233','234'
]

# Create a Validation Split from DS1 (Approx 20% of DS1, patient-wise)
# Ensuring we don't leak patients.
# DS1 has 22 records. 
# Val Set: '223', '230', '209', '208' (Complex arrhythmias are important to validate on)
VAL_RECORDS = ['208', '209', '223', '230']

# Remaining DS1 records are for Training
TRAIN_RECORDS = [rec for rec in DS1_RECORDS if rec not in VAL_RECORDS]

if __name__ == "__main__":
    print(f"Train Records ({len(TRAIN_RECORDS)}): {TRAIN_RECORDS}")
    print(f"Val Records ({len(VAL_RECORDS)}): {VAL_RECORDS}")
    print(f"Test Records ({len(TEST_RECORDS)}): {TEST_RECORDS}")
    
    # Sanity Check Intersection
    assert set(TRAIN_RECORDS).isdisjoint(set(VAL_RECORDS)), "Leakage between Train and Val!"
    assert set(TRAIN_RECORDS).isdisjoint(set(TEST_RECORDS)), "Leakage between Train and Test!"
    assert set(VAL_RECORDS).isdisjoint(set(TEST_RECORDS)), "Leakage between Val and Test!"
    print("ALL SPLITS VALID. NO LEAKAGE.")
