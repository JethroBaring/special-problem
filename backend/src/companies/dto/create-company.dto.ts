import { Prisma } from '@prisma/client';

export class CreateCompanyDto {
  name: string;
  industry?: string | null;
  location?: string | null;
  websiteUrl?: string | null;
  description?: string | null;
  createdAt?: Date | string;
  updatedAt?: Date | string;
  userId: string;
  jobPostings?: Prisma.JobPostingCreateNestedManyWithoutCompanyInput;
}
