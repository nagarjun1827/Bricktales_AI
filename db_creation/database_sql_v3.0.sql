CREATE TABLE "company" (
  "cmp_id" serial PRIMARY KEY,
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
  "project_id" serial PRIMARY KEY,
  "project_name" text,
  "project_code" varchar(50),
  "client_id" int,
  "client_name" text,
  "start_date" date,
  "end_date" date,
  "version" int DEFAULT 1,
  "created_by" text,
  "created_at" timestamptz DEFAULT (now()),
  "updated_by" text,
  "updated_at" timestamptz DEFAULT (now()),
  "project_name_embedding" vector(768),
  "embedding_generated_at" timestamptz DEFAULT (now())
);

CREATE TABLE "locations" (
  "location_id" serial PRIMARY KEY,
  "project_id" int NOT NULL,
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
  "boq_id" serial PRIMARY KEY,
  "project_id" int NOT NULL,
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
  "item_id" serial PRIMARY KEY,
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

CREATE TABLE "to_be_estimated_boq_files" (
  "boq_id" serial PRIMARY KEY,
  "project_id" int NOT NULL,
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

CREATE TABLE "to_be_estimated_boq_items" (
  "item_id" serial PRIMARY KEY,
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

ALTER TABLE "projects" ADD FOREIGN KEY ("client_id") REFERENCES "company" ("cmp_id");

ALTER TABLE "locations" ADD FOREIGN KEY ("project_id") REFERENCES "projects" ("project_id");

ALTER TABLE "store_boq_files" ADD FOREIGN KEY ("project_id") REFERENCES "projects" ("project_id");

ALTER TABLE "store_boq_items" ADD FOREIGN KEY ("boq_id") REFERENCES "store_boq_files" ("boq_id");

ALTER TABLE "store_boq_items" ADD FOREIGN KEY ("location_id") REFERENCES "locations" ("location_id");

ALTER TABLE "to_be_estimated_boq_files" ADD FOREIGN KEY ("project_id") REFERENCES "projects" ("project_id");

ALTER TABLE "to_be_estimated_boq_items" ADD FOREIGN KEY ("boq_id") REFERENCES "to_be_estimated_boq_files" ("boq_id");

ALTER TABLE "to_be_estimated_boq_items" ADD FOREIGN KEY ("location_id") REFERENCES "locations" ("location_id");
