CREATE TABLE "company" (
  "cmp_id" SERIAL PRIMARY KEY,
  "cmp_name" varchar(145) NOT NULL,
  "cmp_addr" varchar(45),
  "fullname" varchar(145),
  "cmp_phone" varchar(45),
  "username" varchar(45),
  "password" varchar(145),
  "access_otp" int,
  "cmp_status" smallint,
  "isenable" smallint,
  "created_at" timestamptz DEFAULT (now()),
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "projects" (
  "project_id" SERIAL PRIMARY KEY,
  "project_name" text,
  "project_code" varchar(50),
  "project_type" varchar(10) NOT NULL,
  "client_id" int,
  "client_name" text,
  "start_date" date,
  "end_date" date,
  "version" int DEFAULT 1,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "tender_projects" (
  "tender_id" SERIAL PRIMARY KEY,
  "project_id" int NOT NULL,
  "tender_number" varchar(100),
  "tender_date" date,
  "submission_deadline" timestamptz,
  "tender_status" varchar(50),
  "tender_value" decimal(18,2),
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "tender_files" (
  "tender_file_id" SERIAL PRIMARY KEY,
  "tender_id" int NOT NULL,
  "file_name" text NOT NULL,
  "file_path" text,
  "file_type" varchar(20),
  "version" int DEFAULT 1,
  "is_active" boolean DEFAULT true,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "store_boq_projects" (
  "store_project_id" SERIAL PRIMARY KEY,
  "project_id" int NOT NULL,
  "store_project_name" text,
  "store_project_code" varchar(50),
  "total_project_value" decimal(18,2),
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "estimate_boq_projects" (
  "estimate_project_id" SERIAL PRIMARY KEY,
  "project_id" int NOT NULL,
  "estimate_project_name" text,
  "estimate_project_code" varchar(50),
  "estimation_status" varchar(50),
  "estimated_value" decimal(18,2),
  "estimated_by" text,
  "estimated_at" timestamptz,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "store_boq_locations" (
  "location_id" SERIAL PRIMARY KEY,
  "store_project_id" int NOT NULL,
  "location_name" text,
  "address" text,
  "latitude" decimal(9,6),
  "longitude" decimal(9,6),
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "estimate_boq_locations" (
  "location_id" SERIAL PRIMARY KEY,
  "estimate_project_id" int NOT NULL,
  "location_name" text,
  "address" text,
  "latitude" decimal(9,6),
  "longitude" decimal(9,6),
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "store_boq_files" (
  "boq_id" SERIAL PRIMARY KEY,
  "store_project_id" int NOT NULL,
  "file_name" text NOT NULL,
  "file_path" text,
  "file_type" varchar(20),
  "version" int DEFAULT 1,
  "is_active" boolean DEFAULT true,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "store_boq_items" (
  "item_id" SERIAL PRIMARY KEY,
  "boq_id" int NOT NULL,
  "item_code" varchar(50),
  "item_description" text NOT NULL,
  "unit_of_measurement" varchar(20) NOT NULL,
  "quantity" decimal(18,3) NOT NULL,
  "supply_unit_rate" decimal(18,2),
  "supply_amount" decimal(18,2),
  "labour_unit_rate" decimal(18,2),
  "labour_amount" decimal(18,2),
  "total_amount" decimal(18,2),
  "location_id" int,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now()),
  "description_embedding" vector(768),
  "embedding_generated_at" timestamptz
);

CREATE TABLE "estimate_boq_files" (
  "boq_id" SERIAL PRIMARY KEY,
  "estimate_project_id" int NOT NULL,
  "file_name" text NOT NULL,
  "file_path" text,
  "file_type" varchar(20),
  "version" int DEFAULT 1,
  "is_active" boolean DEFAULT true,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "estimate_boq_items" (
  "item_id" SERIAL PRIMARY KEY,
  "boq_id" int NOT NULL,
  "item_code" varchar(50),
  "item_description" text NOT NULL,
  "unit_of_measurement" varchar(20) NOT NULL,
  "quantity" decimal(18,3) NOT NULL,
  "location_id" int,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now()),
  "description_embedding" vector(768),
  "embedding_generated_at" timestamptz
);

COMMENT ON COLUMN "projects"."project_type" IS 'CHECK: tender or boq';

ALTER TABLE "projects" ADD FOREIGN KEY ("client_id") REFERENCES "company" ("cmp_id");

ALTER TABLE "tender_projects" ADD FOREIGN KEY ("project_id") REFERENCES "projects" ("project_id");

ALTER TABLE "tender_files" ADD FOREIGN KEY ("tender_id") REFERENCES "tender_projects" ("tender_id");

ALTER TABLE "store_boq_projects" ADD FOREIGN KEY ("project_id") REFERENCES "projects" ("project_id");

ALTER TABLE "estimate_boq_projects" ADD FOREIGN KEY ("project_id") REFERENCES "projects" ("project_id");

ALTER TABLE "store_boq_locations" ADD FOREIGN KEY ("store_project_id") REFERENCES "store_boq_projects" ("store_project_id");

ALTER TABLE "estimate_boq_locations" ADD FOREIGN KEY ("estimate_project_id") REFERENCES "estimate_boq_projects" ("estimate_project_id");

ALTER TABLE "store_boq_files" ADD FOREIGN KEY ("store_project_id") REFERENCES "store_boq_projects" ("store_project_id");

ALTER TABLE "store_boq_items" ADD FOREIGN KEY ("boq_id") REFERENCES "store_boq_files" ("boq_id");

ALTER TABLE "store_boq_items" ADD FOREIGN KEY ("location_id") REFERENCES "store_boq_locations" ("location_id");

ALTER TABLE "estimate_boq_files" ADD FOREIGN KEY ("estimate_project_id") REFERENCES "estimate_boq_projects" ("estimate_project_id");

ALTER TABLE "estimate_boq_items" ADD FOREIGN KEY ("boq_id") REFERENCES "estimate_boq_files" ("boq_id");

ALTER TABLE "estimate_boq_items" ADD FOREIGN KEY ("location_id") REFERENCES "estimate_boq_locations" ("location_id");
