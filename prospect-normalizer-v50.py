#!/usr/bin/env python3
"""
Prospect list normaliser – v52
==============================
20-field master header with smart email selection
Employee range conversion with format variations
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# 1. Master Header - 20 Business Fields
# ---------------------------------------------------------------------------
MASTER_HEADER: List[str] = [
    "Full Name", "First Name", "Last Name", "Linkedin", "Business Email", 
    "Job Title", "Company Name", "Formatted Company Name", "Employees Range", 
    "Company URL", "Company Linkedin", "Industry", "Technology", "City", 
    "Company City", "Company Country", "Description", "Country", "Region", "Keywords"
]

TOKEN_RE = re.compile(r"[^a-z0-9]")
EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"((?:\+\d{1,3})?\s?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9})")

# ---------------------------------------------------------------------------
# 2. Focused Synonyms
# ---------------------------------------------------------------------------
SYNONYMS: Dict[str, str] = {
    # Name fields
    "first": "First Name", "firstname": "First Name", "given_name": "First Name",
    "last": "Last Name", "lastname": "Last Name", "surname": "Last Name",
    "full_name": "Full Name", "contact_full_name": "Full Name", "name": "Full Name",
    
    # Job fields
    "title": "Job Title", "position": "Job Title", "role": "Job Title",
    "job_title": "Job Title", "jobtitle": "Job Title",
    
    # Company fields
    "company": "Company Name", "organization": "Company Name", "org": "Company Name",
    "company_name_cleaned": "Company Name", 
    "formatted_company_name": "Formatted Company Name",
    "website": "Company URL", "domain": "Company URL", "company_website": "Company URL",
    "company_website_domain": "Company URL", "company_url": "Company URL",
    
    # LinkedIn
    "linkedin": "Linkedin", "linkedin_url": "Linkedin", "linkedinurl": "Linkedin",
    "person_linkedin_url": "Linkedin", "contact_li_profile_url": "Linkedin",
    "contactliprofileurl": "Linkedin", "linkedin_profile": "Linkedin",
    "company_linkedin": "Company Linkedin", "company_linkedin_url": "Company Linkedin",
    "company_li_profile_url": "Company Linkedin", "companyliprofileurl": "Company Linkedin",
    "company_linkedin_id": "_skip_", "companylinkedinid": "_skip_",  # Skip ID field
    
    # Location fields
    "contact_city": "City", "contactcity": "City", "location": "City",
    "contact_location_city": "City", "contactlocationcity": "City",
    "company_city": "Company City", "companycity": "Company City",
    "state": "Region", "province": "Region", "contact_state": "Region", 
    "contactstate": "Region", "contact_state_abbr": "Region", "state_abbr": "Region",
    "contact_country": "Country", "contactcountry": "Country", 
    "contact_location_country": "Country", "contactlocationcountry": "Country",
    "company_country": "Company Country", "companycountry": "Company Country",
    
    # Employee
    "employees": "Employees Range", "employee_count": "Employees Range",
    "staff_count": "Employees Range", "company_staff_count": "Employees Range",
    "employees_range": "Employees Range", "company_staff_count_range": "Employees Range",
    "companystaffcountrange": "Employees Range", "employeesrange": "Employees Range",
    "#_employees": "Employees Range", "number_of_employees": "Employees Range",
    "employee_range": "Employees Range", "staffcountrange": "Employees Range",
    
    # Industry/Tech
    "category": "Industry", "sector": "Industry", "vertical": "Industry",
    "company_industry": "Industry", "companyindustry": "Industry",
    "technologies": "Technology", "tech_stack": "Technology",
    
    # Description
    "company_description": "Description", "about": "Description", "summary": "Description",
    
    # Keywords
    "tags": "Keywords", "specialties": "Keywords",
}

# Email validation priority (higher is better)
EMAIL_VALIDATION_PRIORITY = {
    "valid": 3,
    "accept-all": 2,
    "accept_all": 2,
    "acceptall": 2,
    "unknown": 1,
    "invalid": 0,
}

# ---------------------------------------------------------------------------
# 3. Employee Count to Range Converter
# ---------------------------------------------------------------------------
def convert_to_employee_range(value) -> str:
    """Convert numeric employee counts to standard ranges"""
    if pd.isna(value) or value == "":
        return pd.NA
    
    if isinstance(value, str):
        val_str = str(value).strip()
        
        # Map common formats to our standard
        range_mappings = {
            # With spaces and "employees"
            "1 - 10 employees": "1-10", "2 - 10 employees": "1-10",
            "11 - 50 employees": "11-50",
            "51 - 200 employees": "51-200",
            "201 - 500 employees": "201-500",
            "501 - 1000 employees": "501-1000", "501 - 1,000 employees": "501-1000",
            "1001 - 5000 employees": "1001-5000", "1,001 - 5,000 employees": "1001-5000",
            "5001 - 10000 employees": "5001-10000", "5,001 - 10,000 employees": "5001-10000",
            # Without "employees"
            "1-10": "1-10", "11-50": "11-50", "51-200": "51-200",
            "201-500": "201-500", "501-1000": "501-1000",
            "1001-5000": "1001-5000", "5001-10000": "5001-10000",
            "10000+": "10000+", "10,000+": "10000+",
        }
        
        # Direct mapping
        if val_str in range_mappings:
            return range_mappings[val_str]
        
        # Try to extract range pattern
        range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', val_str)
        if range_match:
            low = int(range_match.group(1))
            high = int(range_match.group(2).replace(',', ''))
            
            # Map to standard ranges based on high value
            if high <= 10:
                return "1-10"
            elif high <= 50:
                return "11-50"
            elif high <= 200:
                return "51-200"
            elif high <= 500:
                return "201-500"
            elif high <= 1000:
                return "501-1000"
            elif high <= 5000:
                return "1001-5000"
            elif high <= 10000:
                return "5001-10000"
            else:
                return "10000+"
        
        # Extract single number
        nums = re.findall(r'\d+', val_str.replace(',', ''))
        if nums:
            value = int(nums[0])
        else:
            return pd.NA
    
    try:
        num = int(value)
        if num <= 10:
            return "1-10"
        elif num <= 50:
            return "11-50"
        elif num <= 200:
            return "51-200"
        elif num <= 500:
            return "201-500"
        elif num <= 1000:
            return "501-1000"
        elif num <= 5000:
            return "1001-5000"
        elif num <= 10000:
            return "5001-10000"
        else:
            return "10000+"
    except:
        return pd.NA

# ---------------------------------------------------------------------------
# 4. Email Selection Logic
# ---------------------------------------------------------------------------
def find_best_email(row: pd.Series, email_cols: List[str], validation_cols: Dict[str, str]) -> Optional[str]:
    """Find the best email from multiple columns based on validation status"""
    best_email = None
    best_priority = -1
    
    for email_col in email_cols:
        if email_col not in row.index:
            continue
            
        email = row[email_col]
        if pd.isna(email) or email == "":
            continue
            
        # Clean the email
        email_match = EMAIL_RE.search(str(email))
        if not email_match:
            continue
        email = email_match.group(1).lower()
        
        # Check validation status if available
        validation_col = validation_cols.get(email_col)
        if validation_col and validation_col in row.index:
            validation_status = str(row[validation_col]).lower()
            priority = EMAIL_VALIDATION_PRIORITY.get(validation_status, 0)
        else:
            # No validation info, treat as unknown
            priority = 1
            
        if priority > best_priority:
            best_email = email
            best_priority = priority
            
        # Stop if we found a valid email
        if best_priority >= 3:
            break
            
    return best_email

# ---------------------------------------------------------------------------
# 5. Column Mapping
# ---------------------------------------------------------------------------
def normalize_token(s: str) -> str:
    return TOKEN_RE.sub("", s.lower())

def identify_email_columns(cols: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Identify email columns and their validation columns"""
    email_cols = []
    validation_map = {}
    
    for col in cols:
        tok = normalize_token(col)
        
        # Main email columns
        if re.match(r"^email\d*$", tok) or tok in ["primaryemail", "businessemail", "workemail"]:
            email_cols.append(col)
            
            # Look for corresponding validation column
            if re.match(r"^email\d+$", tok):
                num = re.findall(r"\d+", tok)[0]
                for vcol in cols:
                    vtok = normalize_token(vcol)
                    if vtok == f"email{num}validation":
                        validation_map[col] = vcol
                        break
                        
    return email_cols, validation_map

def auto_map_columns(cols: Iterable[str], df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    mapping: Dict[str, str] = {}
    unmapped: List[str] = []
    canonical = {normalize_token(h): h for h in MASTER_HEADER}
    mapped_targets = set()
    
    # Get all columns as list
    cols_list = list(cols)
    
    # Identify email columns for special handling
    email_cols, validation_map = identify_email_columns(cols_list)

    for col in cols_list:
        tok = normalize_token(col)
        
        # Skip validation columns
        if "validation" in tok or "totalai" in tok:
            continue
            
        # Skip if already an email column (handled separately)
        if col in email_cols:
            continue
        
        # Exact match
        if tok in canonical and canonical[tok] not in mapped_targets:
            mapping[col] = canonical[tok]
            mapped_targets.add(canonical[tok])
            continue
            
        # Synonym match
        if tok in SYNONYMS:
            if SYNONYMS[tok] not in mapped_targets:
                mapping[col] = SYNONYMS[tok]
                mapped_targets.add(SYNONYMS[tok])
            continue
            
        # Fuzzy match only for longer names
        if len(tok) >= 5:
            cand, score, _ = process.extractOne(tok, canonical.keys(), scorer=fuzz.ratio) or (None, 0, None)
            if score >= 90 and canonical[cand] not in mapped_targets:
                mapping[col] = canonical[cand]
                mapped_targets.add(canonical[cand])
            else:
                unmapped.append(col)
        else:
            unmapped.append(col)
    
    # Store email column info for later processing
    mapping["_email_cols"] = email_cols
    mapping["_validation_map"] = validation_map
            
    return mapping, unmapped

# ---------------------------------------------------------------------------
# 6. Frame operations
# ---------------------------------------------------------------------------
def normalize_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    ren_map, unmapped = auto_map_columns(df.columns, df)
    
    # Extract special mappings
    email_cols = ren_map.pop("_email_cols", [])
    validation_map = ren_map.pop("_validation_map", {})
    
    # Rename columns - this may create duplicates
    df = df.rename(columns=ren_map)
    
    # Collapse duplicate columns BEFORE other processing
    for col in MASTER_HEADER:
        if col in df.columns:
            dupes = df.loc[:, df.columns == col]
            if dupes.shape[1] > 1:
                # Keep first non-null value across duplicates
                df = df.drop(columns=[c for c in df.columns if c == col])
                df[col] = dupes.bfill(axis=1).iloc[:, 0]
    
    # Handle email selection
    if email_cols and "Business Email" not in df:
        df["Business Email"] = df.apply(lambda row: find_best_email(row, email_cols, validation_map), axis=1)
    elif "Business Email" in df:
        # Clean existing business email
        df["Business Email"] = df["Business Email"].astype(str).str.extract(EMAIL_RE, expand=False).str.strip().str.lower()
        df["Business Email"] = df["Business Email"].replace("", pd.NA)
        
    # Convert employee counts
    if "Employees Range" in df:
        df["Employees Range"] = df["Employees Range"].apply(convert_to_employee_range)
    
    # Generate Full Name if missing
    if "Full Name" not in df and "First Name" in df and "Last Name" in df:
        df["Full Name"] = df["First Name"].fillna("") + " " + df["Last Name"].fillna("")
        df["Full Name"] = df["Full Name"].str.strip().replace("", pd.NA)
    
    # Ensure all master columns exist
    for col in MASTER_HEADER:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Select columns in the correct order
    result_df = df[MASTER_HEADER].copy()
            
    return result_df, ren_map, unmapped

# ---------------------------------------------------------------------------
# 7. I/O helpers
# ---------------------------------------------------------------------------
def read_any(path: Path, chunksize: int | None):
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        yield pd.read_excel(path)
    elif suf == ".csv":
        if chunksize:
            yield from pd.read_csv(path, chunksize=chunksize)
        else:
            yield pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def process_file(path: Path, out_dir: Path, chunksize: int | None, quiet: bool):
    frames, mapping_acc, unmapped_acc = [], {}, set()
    
    try:
        for chunk in read_any(path, chunksize):
            norm, mapping, unmapped = normalize_frame(chunk)
            frames.append(norm)
            if isinstance(mapping, dict):
                mapping_acc.update({k: v for k, v in mapping.items() if not k.startswith("_")})
            if isinstance(unmapped, list):
                unmapped_acc.update(unmapped)
            
        out_df = pd.concat(frames, ignore_index=True)
        
        # Remove rows with no email or company
        out_df = out_df.dropna(subset=["Business Email", "Company Name"], how="all")
        
        out_file = out_dir / f"normalized_{path.stem}.csv"
        out_df.to_csv(out_file, index=False)
        
        if not quiet:
            print(f"✔ {path.name} → {out_file.name}  (rows={len(out_df):,})")
            if unmapped_acc:
                print(f"   ⚠ Unmapped columns ({len(unmapped_acc)}): {', '.join(sorted(unmapped_acc)[:5])}")
                if len(unmapped_acc) > 5:
                    print(f"     ... and {len(unmapped_acc)-5} more")
    except Exception as e:
        import traceback
        print(f"✖ Failed processing {path}: {e}")
        if not quiet:
            print(f"   Traceback: {traceback.format_exc()}")
        raise

# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser("Normalize prospect lists to simplified 15-field format")
    ap.add_argument("paths", nargs="+", help="CSV/XLSX files or directories")
    ap.add_argument("--out", default="./normalized_output", help="Output folder")
    ap.add_argument("--chunksize", type=int, default=0, help="Chunk size for large CSVs (0 = off)")
    ap.add_argument("--quiet", action="store_true", help="Suppress warnings and logs")
    args = ap.parse_args(argv)

    if args.quiet:
        warnings.filterwarnings("ignore")

    in_files: List[Path] = []
    for p in args.paths:
        pth = Path(p)
        if pth.is_dir():
            in_files.extend(list(pth.glob("*.csv")) + list(pth.glob("*.xls*")))
        elif pth.is_file():
            in_files.append(pth)
        else:
            print(f"Skipping unknown path {p}", file=sys.stderr)
            
    if not in_files:
        sys.exit("No valid input files found")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    
    for f in in_files:
        try:
            process_file(f, out_dir, args.chunksize or None, args.quiet)
        except Exception as exc:
            print(f"✖ Failed processing {f}: {exc}", file=sys.stderr)

if __name__ == "__main__":
    main()