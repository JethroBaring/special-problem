import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const main = async () => {
  const jobCategories = [
    'Software Development',
    'Artificial Intelligence',
    'Machine Learning',
    'Data Science',
    'Cybersecurity',
    'Cloud Computing',
    'Blockchain',
    'Game Development',
    'DevOps & Site Reliability',
    'Mobile Development',
    'Web Development',
    'Frontend Development',
    'Backend Development',
    'Full Stack Development',
    'Quality Assurance & Testing',
    'UI/UX Design',
    'Product Management',
    'Marketing',
    'Finance & Accounting',
    'Human Resources',
    'Legal & Compliance',
    'Healthcare & Medical',
    'Education & Teaching',
    'Engineering',
    'Electrical Engineering',
    'Mechanical Engineering',
    'Civil Engineering',
    'Automotive Engineering',
    'Supply Chain & Logistics',
    'Sales & Business Development',
  ];

  for (const category of jobCategories) {
    await prisma.jobCategory.upsert({
      where: { name: category },
      update: {},
      create: { name: category },
    });
  }

  console.log('✅ Job categories seeded successfully.');
};

main()
  .then(async () => {
    await prisma.$disconnect();
  })

  .catch(async (e) => {
    console.error(e);

    await prisma.$disconnect();

    process.exit(1);
  });
